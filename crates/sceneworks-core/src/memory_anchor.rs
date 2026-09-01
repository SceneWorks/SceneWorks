//! Measured memory anchors and the analytic peak derivation built on them (sc-22507, epic 22505).
//!
//! One anchor per `(model, quant tier, backend lane)` carries the measured per-phase peak
//! decomposition of a single retained calibration render. Everything else is derived analytically:
//! activation terms scale linearly in latent tokens (`area x latent frames`), decode scales in
//! output voxels, attention transients are bounded by the declared rung parameters, and bounded
//! regimes (tiled decode, windowed transformer residency, deferred materialization) substitute
//! architecture-bounded terms for the anchored ones. This replaces grid measurement: a request at
//! a `(geometry, frames)` cell that was never measured is admitted from the derived estimate.
//!
//! The store is checked in at `config/memory-anchors.json` and is validated against the immutable
//! retained evidence it was extracted from (`PACKAGED_MEMORY_ANCHOR_SOURCES`): every anchor's
//! source file must hash to the recorded digest and the anchor's phase bytes must equal the source
//! record's measured values byte-for-byte, so the store cannot drift from the corpus it cites.
//!
//! Like `video_memory_curves`, this module deliberately has no `gen-core` dependency; worker lanes
//! translate their own backend/rung/load-shape types at the edge.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MEMORY_ANCHOR_SCHEMA_VERSION: u32 = 1;

/// The checked-in anchor store.
pub const PACKAGED_MEMORY_ANCHORS: &str = include_str!("../../../config/memory-anchors.json");

/// Immutable retained evidence the anchors were extracted from. Every anchor's `source.path` must
/// name one of these files, whose bytes are compiled in so the handshake cannot be bypassed by
/// editing the file on disk.
const PACKAGED_MEMORY_ANCHOR_SOURCES: &[(&str, &str)] = &[(
    "docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json",
    include_str!("../../../docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json"),
)];

// ---------------------------------------------------------------------------------------------
// Architecture facts (LTX-2.5 pipeline geometry).
// ---------------------------------------------------------------------------------------------

/// LTX's video VAE is x32 spatial: one latent token per 32x32 output patch. Mirrors the constant
/// the MLX calibration adapter derives its regressors from
/// (`crates/sceneworks-memory-adapter/src/bin/mlx.rs`).
pub const LTX_LATENT_PATCH_EDGE_PX: u64 = 32;

/// LTX's video VAE is x8 causal temporal: `latent_frames = 1 + ceil((frames - 1) / 8)`. Same
/// source as [`LTX_LATENT_PATCH_EDGE_PX`]; the engine's frame lattice is `frames = 1 + 8k`.
pub const LTX_TEMPORAL_SCALE: u64 = 8;

// ---------------------------------------------------------------------------------------------
// Derivation coefficients. Each names the architecture term it prices; each uncertainty a
// coefficient cannot pin exactly is covered by [`ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`], and the
// corpus validation test (`derivation_brackets_every_retained_corpus_peak`) is the falsifier.
// ---------------------------------------------------------------------------------------------

/// Conditioning-phase bytes per latent token: the packed video latent plus the Gemma text
/// cross-attention context held in fp32 workspace (4096 f32 lanes x 8 concurrently live per-token
/// tensors). Retained-corpus slopes across tiers/load shapes sit at 110-131 KiB/token.
pub const COND_PER_TOKEN_BYTES: i128 = 131_072;

/// Conditioning-phase bound under deferred materialization: only the projected text embeddings and
/// latent init are resident (the transformer stays unmaterialized), which the retained captures
/// show as geometry-independent (11.73 GB active / 12.02 GB allocator at every measured geometry).
/// Bound set above the allocator figure.
pub const COND_DEFERRED_BOUND_BYTES: u64 = 12_900_000_000;

/// Denoise-phase bytes per latent token: the DiT forward's live activation set (residual stream,
/// attention chunk workspace under the declared `attentionChunkSize`, MLP intermediates) at fp32.
/// Retained-corpus slopes sit at 306-330 KiB/token across all three tiers and both load shapes.
pub const DENOISE_PER_TOKEN_BYTES: i128 = 335_872;

/// Denoise-phase intercept under `bounded_transformer_residency` (window size 1): one resident
/// AvDiT block of the declared 48 plus the bounded attention workspace, replacing the full
/// transformer residency the anchor was measured with. The per-token activation slope is
/// unchanged by windowing ([`DENOISE_PER_TOKEN_BYTES`]).
pub const DENOISE_WINDOWED_BASE_BYTES: u64 = 2_650_000_000;

/// Decode-phase bytes per output voxel in the single-pass conv-VAE regime: concurrently live
/// pixel-space working copies (four fp32 RGB-equivalent planes). Retained-corpus slopes span
/// 30-63 B/voxel; the spread is covered by [`ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`].
pub const DECODE_PER_VOXEL_BYTES: i128 = 48;

/// Decode-phase bound under `bounded_decode` (declared `decodeTileEdge`/`decodeOverlap`): decoder
/// weights plus tile-bounded workspace, independent of output geometry. Retained tiled captures
/// peak at 8.26 GB across tiers; bound set above them.
pub const DECODE_TILED_BOUND_BYTES: u64 = 9_000_000_000;

/// Multiplicative margin applied to every derived phase peak. It covers, jointly: the MLX
/// allocator envelope above the phase active peak (observed up to 15.9% in the retained corpus —
/// cache retention across phase transitions), and the residual spread of the per-token /
/// per-voxel coefficients around the architecture-derived values above (observed under 4%).
pub const ANCHOR_ALLOCATOR_ENVELOPE_MARGIN: f64 = 0.17;

/// Validation-only tightness budget: the corpus validation test refuses a derived bound more than
/// this fraction above the measured allocator envelope, so the margins above stay falsifiable
/// instead of quietly widening into a vacuous always-passes bound.
pub const ANCHOR_VALIDATION_TIGHTNESS_BUDGET: f64 = 0.25;

// ---------------------------------------------------------------------------------------------
// Store schema.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryAnchorStore {
    pub schema_version: u32,
    pub anchors: Vec<MemoryAnchor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorBackend {
    Mlx,
    Candle,
}

impl AnchorBackend {
    pub const fn as_key(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::Candle => "candle",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorLoadShape {
    EagerMaterialization,
    DeferredMaterialization,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnchorSource {
    /// Repo-relative path of the retained evidence file; must be a compiled-in source.
    pub path: String,
    /// SHA-256 of that file's bytes at extraction time.
    pub sha256: String,
    /// The calibration record inside it this anchor was extracted from.
    pub record_id: String,
    pub calibration_fingerprint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnchorGeometry {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
}

/// The measured per-phase ACTIVE peak decomposition of the anchor render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnchorPhaseBytes {
    pub conditioning: u64,
    pub denoise: u64,
    pub decode: u64,
}

/// One measured anchor: identity plus the peak decomposition of exactly one retained render.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryAnchor {
    pub id: String,
    pub model_id: String,
    pub model_family: String,
    pub backend: AnchorBackend,
    /// Plan tier key: `q4` / `q8` / `bf16` / ...
    pub tier: String,
    pub mode: String,
    /// The anchor was measured overlay-free; a future overlay anchor is a new row, never a reuse.
    pub overlay: Option<String>,
    pub reference_count: u32,
    /// Materialization shape the anchor render ran under. The derivation reaches the OTHER shape
    /// analytically (see [`COND_DEFERRED_BOUND_BYTES`]), so this stays informational identity.
    pub load_shape: AnchorLoadShape,
    pub source: AnchorSource,
    pub geometry: AnchorGeometry,
    pub phase_active_peak_bytes: AnchorPhaseBytes,
    /// The measured overall allocator envelope of the anchor render (active + reclaimable).
    pub overall_allocator_envelope_bytes: u64,
}

/// Strict parse plus the invariants serde cannot express. No partial store is usable.
pub fn load_memory_anchors(raw: &str) -> Result<MemoryAnchorStore, String> {
    let store: MemoryAnchorStore = serde_json::from_str(raw)
        .map_err(|error| format!("memory anchors do not parse: {error}"))?;
    if store.schema_version != MEMORY_ANCHOR_SCHEMA_VERSION {
        return Err(format!(
            "memory anchor schema version {} is not the supported {MEMORY_ANCHOR_SCHEMA_VERSION}",
            store.schema_version
        ));
    }
    let mut identities = BTreeMap::new();
    for anchor in &store.anchors {
        let identity = (anchor.model_id.clone(), anchor.tier.clone(), anchor.backend);
        if let Some(previous) = identities.insert(identity, &anchor.id) {
            return Err(format!(
                "duplicate memory anchor for ({}, {}, {}): {} and {} — exactly one anchor per \
                 (model, tier, backend lane) is the schema invariant",
                anchor.model_id,
                anchor.tier,
                anchor.backend.as_key(),
                previous,
                anchor.id
            ));
        }
        validate_anchor(anchor)?;
    }
    Ok(store)
}

/// One anchor's extraction handshake against the compiled-in retained evidence.
fn validate_anchor(anchor: &MemoryAnchor) -> Result<(), String> {
    if anchor.geometry.width == 0 || anchor.geometry.height == 0 || anchor.geometry.frames == 0 {
        return Err(format!(
            "memory anchor {} has a degenerate geometry",
            anchor.id
        ));
    }
    if anchor.phase_active_peak_bytes.conditioning == 0
        || anchor.phase_active_peak_bytes.denoise == 0
        || anchor.phase_active_peak_bytes.decode == 0
        || anchor.overall_allocator_envelope_bytes == 0
    {
        return Err(format!("memory anchor {} has a zero phase peak", anchor.id));
    }
    let Some((_, source_raw)) = PACKAGED_MEMORY_ANCHOR_SOURCES
        .iter()
        .find(|(path, _)| *path == anchor.source.path)
    else {
        return Err(format!(
            "memory anchor {} cites source {} which is not a compiled retained-evidence file",
            anchor.id, anchor.source.path
        ));
    };
    let digest = format!("{:x}", Sha256::digest(source_raw.as_bytes()));
    if digest != anchor.source.sha256 {
        return Err(format!(
            "memory anchor {} source digest mismatch for {}: recorded {} actual {digest}",
            anchor.id, anchor.source.path, anchor.source.sha256
        ));
    }
    let source: serde_json::Value = serde_json::from_str(source_raw).map_err(|error| {
        format!(
            "retained evidence {} does not parse: {error}",
            anchor.source.path
        )
    })?;
    let records = source
        .get("records")
        .and_then(|records| records.as_array())
        .ok_or_else(|| format!("retained evidence {} has no records", anchor.source.path))?;
    let record = records
        .iter()
        .find(|record| {
            record.get("id").and_then(|id| id.as_str()) == Some(anchor.source.record_id.as_str())
        })
        .ok_or_else(|| {
            format!(
                "memory anchor {} cites record {} absent from {}",
                anchor.id, anchor.source.record_id, anchor.source.path
            )
        })?;
    let target = record.get("target").unwrap_or(&serde_json::Value::Null);
    let str_at = |value: &serde_json::Value, key: &str| {
        value
            .get(key)
            .and_then(|value| value.as_str())
            .map(str::to_owned)
    };
    if str_at(target, "modelId").as_deref() != Some(anchor.model_id.as_str())
        || str_at(target, "tier").as_deref() != Some(anchor.tier.as_str())
        || str_at(record, "backend").as_deref() != Some(anchor.backend.as_key())
        || str_at(target, "mode").as_deref() != Some(anchor.mode.as_str())
        || str_at(record, "calibrationFingerprint").as_deref()
            != Some(anchor.source.calibration_fingerprint.as_str())
    {
        return Err(format!(
            "memory anchor {} identity disagrees with its source record {}",
            anchor.id, anchor.source.record_id
        ));
    }
    let geometry = target.get("geometry").unwrap_or(&serde_json::Value::Null);
    let u64_at = |value: &serde_json::Value, key: &str| value.get(key).and_then(|v| v.as_u64());
    if u64_at(geometry, "width") != Some(u64::from(anchor.geometry.width))
        || u64_at(geometry, "height") != Some(u64::from(anchor.geometry.height))
        || u64_at(geometry, "frames") != Some(u64::from(anchor.geometry.frames))
    {
        return Err(format!(
            "memory anchor {} geometry disagrees with its source record {}",
            anchor.id, anchor.source.record_id
        ));
    }
    let mut measured = BTreeMap::new();
    if let Some(measurements) = record
        .get("diagnostics")
        .and_then(|diagnostics| diagnostics.get("measurements"))
        .and_then(|measurements| measurements.as_array())
    {
        for entry in measurements {
            if let (Some(name), Some(value)) = (
                entry.get("name").and_then(|name| name.as_str()),
                entry.get("value").and_then(|value| value.as_u64()),
            ) {
                measured.insert(name.to_owned(), value);
            }
        }
    }
    let envelope = record
        .get("observedMemory")
        .and_then(|memory| memory.get("overall"))
        .and_then(|overall| overall.get("allocatorBytes"))
        .and_then(|bytes| bytes.as_u64());
    if measured.get("conditioningActivePeak").copied()
        != Some(anchor.phase_active_peak_bytes.conditioning)
        || measured.get("denoiseActivePeak").copied()
            != Some(anchor.phase_active_peak_bytes.denoise)
        || measured.get("decodeActivePeak").copied() != Some(anchor.phase_active_peak_bytes.decode)
        || envelope != Some(anchor.overall_allocator_envelope_bytes)
    {
        return Err(format!(
            "memory anchor {} peak bytes disagree with its source record {} — the store may not \
             drift from the retained evidence it cites",
            anchor.id, anchor.source.record_id
        ));
    }
    Ok(())
}

/// The packaged store, parsed and validated once. `None` is fail-open: callers keep their
/// pre-existing floor.
pub fn packaged_memory_anchors() -> Option<&'static MemoryAnchorStore> {
    static PACKAGED: OnceLock<Option<MemoryAnchorStore>> = OnceLock::new();
    PACKAGED
        .get_or_init(|| load_memory_anchors(PACKAGED_MEMORY_ANCHORS).ok())
        .as_ref()
}

impl MemoryAnchorStore {
    /// The unique anchor for one `(model, backend lane, tier)` coordinate.
    pub fn anchor_for(
        &self,
        model_id: &str,
        backend: AnchorBackend,
        tier: &str,
    ) -> Option<&MemoryAnchor> {
        self.anchors.iter().find(|anchor| {
            anchor.model_id == model_id && anchor.backend == backend && anchor.tier == tier
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Derivation.
// ---------------------------------------------------------------------------------------------

/// The workload axes the derivation prices. Geometry is the requested output; the three regime
/// flags translate the engaged rung composition and materialization shape of the candidate being
/// graded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorDeriveRequest {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    /// `bounded_decode` engaged: decode is tile-bounded ([`DECODE_TILED_BOUND_BYTES`]).
    pub decode_tiled: bool,
    /// `bounded_transformer_residency` engaged: denoise holds one window instead of the full
    /// transformer ([`DENOISE_WINDOWED_BASE_BYTES`]).
    pub transformer_windowed: bool,
    /// Deferred materialization: conditioning holds no transformer weights
    /// ([`COND_DEFERRED_BOUND_BYTES`]).
    pub deferred_materialization: bool,
}

/// Margin-widened per-phase peak estimates. The admission peak is the max over phases; the shared
/// selector's backend estimate margin still rides on top, exactly as it does for fitted curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorDerivedPhases {
    pub conditioning: u64,
    pub denoise: u64,
    pub decode: u64,
}

impl AnchorDerivedPhases {
    pub const fn peak_bytes(self) -> u64 {
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

/// Latent token count: spatial patches x latent temporal depth (ceil mappings so an off-lattice
/// request rounds up, never down).
fn latent_tokens(width: u32, height: u32, frames: u32) -> i128 {
    let patches_w = (u64::from(width)).div_ceil(LTX_LATENT_PATCH_EDGE_PX);
    let patches_h = (u64::from(height)).div_ceil(LTX_LATENT_PATCH_EDGE_PX);
    let latent_frames = 1 + (u64::from(frames) - 1).div_ceil(LTX_TEMPORAL_SCALE);
    i128::from(patches_w) * i128::from(patches_h) * i128::from(latent_frames)
}

fn voxels(width: u32, height: u32, frames: u32) -> i128 {
    i128::from(width) * i128::from(height) * i128::from(frames)
}

/// Widen one derived phase estimate by [`ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`] in integer bytes.
fn widened(bytes: i128) -> Option<u64> {
    let bytes = u64::try_from(bytes.max(0)).ok()?;
    let widened = (bytes as f64 * (1.0 + ANCHOR_ALLOCATOR_ENVELOPE_MARGIN)).ceil();
    (widened.is_finite() && widened < u64::MAX as f64).then_some(widened as u64)
}

impl MemoryAnchor {
    /// Derive the per-phase peak estimate for one requested video workload from this anchor.
    ///
    /// * conditioning: anchored intercept + [`COND_PER_TOKEN_BYTES`] per latent token, or the
    ///   deferred-materialization bound.
    /// * denoise: anchored intercept + [`DENOISE_PER_TOKEN_BYTES`] per latent token (attention
    ///   transient bounded by the declared chunk parameter, so no super-linear term), or the
    ///   windowed-residency intercept plus the same slope.
    /// * decode: anchored intercept + [`DECODE_PER_VOXEL_BYTES`] per output voxel, or the
    ///   tile-bounded constant.
    ///
    /// Returns `None` on degenerate geometry; every estimate is widened by
    /// [`ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`].
    pub fn derive_video_phase_peaks(
        &self,
        request: AnchorDeriveRequest,
    ) -> Option<AnchorDerivedPhases> {
        if request.width == 0 || request.height == 0 || request.frames == 0 {
            return None;
        }
        let anchor_tokens = latent_tokens(
            self.geometry.width,
            self.geometry.height,
            self.geometry.frames,
        );
        let anchor_voxels = voxels(
            self.geometry.width,
            self.geometry.height,
            self.geometry.frames,
        );
        let tokens = latent_tokens(request.width, request.height, request.frames);
        let voxels = voxels(request.width, request.height, request.frames);

        let conditioning = if request.deferred_materialization {
            i128::from(COND_DEFERRED_BOUND_BYTES)
        } else {
            i128::from(self.phase_active_peak_bytes.conditioning)
                + COND_PER_TOKEN_BYTES * (tokens - anchor_tokens)
        };
        let denoise = if request.transformer_windowed {
            i128::from(DENOISE_WINDOWED_BASE_BYTES) + DENOISE_PER_TOKEN_BYTES * tokens
        } else {
            i128::from(self.phase_active_peak_bytes.denoise)
                + DENOISE_PER_TOKEN_BYTES * (tokens - anchor_tokens)
        };
        let decode = if request.decode_tiled {
            i128::from(DECODE_TILED_BOUND_BYTES)
        } else {
            i128::from(self.phase_active_peak_bytes.decode)
                + DECODE_PER_VOXEL_BYTES * (voxels - anchor_voxels)
        };
        Some(AnchorDerivedPhases {
            conditioning: widened(conditioning)?,
            denoise: widened(denoise)?,
            decode: widened(decode)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LTX25_CORPUS_PATH: &str = "docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json";

    fn corpus_raw() -> &'static str {
        PACKAGED_MEMORY_ANCHOR_SOURCES
            .iter()
            .find(|(path, _)| *path == LTX25_CORPUS_PATH)
            .map(|(_, raw)| *raw)
            .expect("the LTX-2.5 MLX retained corpus is compiled in")
    }

    struct CorpusRecord {
        id: String,
        tier: String,
        width: u32,
        height: u32,
        frames: u32,
        decode_tiled: bool,
        transformer_windowed: bool,
        deferred: bool,
        conditioning_active: u64,
        denoise_active: u64,
        decode_active: u64,
        overall_envelope: u64,
    }

    /// Every retained, runtime-complete LTX-2.5 MLX measured record — the validation corpus.
    fn retained_corpus() -> Vec<CorpusRecord> {
        let source: serde_json::Value =
            serde_json::from_str(corpus_raw()).expect("retained corpus parses");
        let records = source["records"].as_array().expect("corpus records");
        records
            .iter()
            .filter(|record| {
                record["target"]["modelId"].as_str() == Some("ltx_2_5")
                    && record["backend"].as_str() == Some("mlx")
                    && record["status"].as_str() == Some("runtime_complete")
            })
            .map(|record| {
                let geometry = &record["target"]["geometry"];
                let engaged: Vec<&str> = record["strategy"]["engagedRungs"]
                    .as_array()
                    .expect("engaged rungs")
                    .iter()
                    .filter_map(|value| value.as_str())
                    .collect();
                let measured: std::collections::BTreeMap<&str, u64> = record["diagnostics"]
                    ["measurements"]
                    .as_array()
                    .expect("measurements")
                    .iter()
                    .filter_map(|entry| Some((entry["name"].as_str()?, entry["value"].as_u64()?)))
                    .collect();
                CorpusRecord {
                    id: record["id"].as_str().expect("record id").to_owned(),
                    tier: record["target"]["tier"].as_str().expect("tier").to_owned(),
                    width: geometry["width"].as_u64().expect("width") as u32,
                    height: geometry["height"].as_u64().expect("height") as u32,
                    frames: geometry["frames"].as_u64().expect("frames") as u32,
                    decode_tiled: engaged.contains(&"bounded_decode"),
                    transformer_windowed: engaged.contains(&"bounded_transformer_residency"),
                    deferred: record["loadShape"].as_str() == Some("deferred_materialization"),
                    conditioning_active: measured["conditioningActivePeak"],
                    denoise_active: measured["denoiseActivePeak"],
                    decode_active: measured["decodeActivePeak"],
                    overall_envelope: record["observedMemory"]["overall"]["allocatorBytes"]
                        .as_u64()
                        .expect("overall envelope"),
                }
            })
            .collect()
    }

    fn store() -> &'static MemoryAnchorStore {
        packaged_memory_anchors().expect("the packaged anchor store loads")
    }

    // -------------------------------------------------------------------------------------
    // AC 1 (store half): LTX-2.5 MLX q4/q8/bf16 each carry exactly one anchor.
    // -------------------------------------------------------------------------------------

    #[test]
    fn ltx25_mlx_carries_exactly_one_anchor_per_tier() {
        for tier in ["q4", "q8", "bf16"] {
            let matching = store()
                .anchors
                .iter()
                .filter(|anchor| {
                    anchor.model_id == "ltx_2_5"
                        && anchor.backend == AnchorBackend::Mlx
                        && anchor.tier == tier
                })
                .count();
            assert_eq!(matching, 1, "tier {tier} must carry exactly one anchor");
            let anchor = store()
                .anchor_for("ltx_2_5", AnchorBackend::Mlx, tier)
                .expect("lookup resolves the tier's anchor");
            assert_eq!(anchor.tier, tier);
            assert_eq!(anchor.source.path, LTX25_CORPUS_PATH);
        }
    }

    #[test]
    fn a_duplicate_anchor_identity_is_rejected() {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        let clone = doctored["anchors"][0].clone();
        doctored["anchors"]
            .as_array_mut()
            .expect("anchors")
            .push(clone);
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("a duplicated (model, tier, backend) must be rejected");
        assert!(error.contains("duplicate memory anchor"), "{error}");
    }

    #[test]
    fn an_anchor_that_drifts_from_its_source_record_is_rejected() {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        let peak = doctored["anchors"][0]["phaseActivePeakBytes"]["conditioning"]
            .as_u64()
            .expect("conditioning peak");
        doctored["anchors"][0]["phaseActivePeakBytes"]["conditioning"] =
            serde_json::json!(peak + 1);
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("a store whose bytes drift from the retained evidence must be rejected");
        assert!(error.contains("disagree"), "{error}");
    }

    #[test]
    fn a_stale_source_digest_or_foreign_source_path_is_rejected() {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][0]["source"]["sha256"] = serde_json::json!("0".repeat(64));
        let error = load_memory_anchors(&doctored.to_string()).expect_err("digest must bind");
        assert!(error.contains("digest mismatch"), "{error}");

        let mut foreign: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        foreign["anchors"][0]["source"]["path"] = serde_json::json!("docs/nowhere.json");
        let error = load_memory_anchors(&foreign.to_string()).expect_err("path must be compiled");
        assert!(
            error.contains("not a compiled retained-evidence file"),
            "{error}"
        );
    }

    // -------------------------------------------------------------------------------------
    // Derivation properties.
    // -------------------------------------------------------------------------------------

    fn plain_request(width: u32, height: u32, frames: u32) -> AnchorDeriveRequest {
        AnchorDeriveRequest {
            width,
            height,
            frames,
            decode_tiled: false,
            transformer_windowed: false,
            deferred_materialization: false,
        }
    }

    #[test]
    fn derivation_at_the_anchor_geometry_reproduces_the_anchor_peaks_plus_margin() {
        let anchor = store()
            .anchor_for("ltx_2_5", AnchorBackend::Mlx, "q8")
            .expect("q8 anchor");
        let derived = anchor
            .derive_video_phase_peaks(plain_request(
                anchor.geometry.width,
                anchor.geometry.height,
                anchor.geometry.frames,
            ))
            .expect("derivable at the anchor's own geometry");
        let expect = |measured: u64| {
            ((measured as f64) * (1.0 + ANCHOR_ALLOCATOR_ENVELOPE_MARGIN)).ceil() as u64
        };
        assert_eq!(
            derived.conditioning,
            expect(anchor.phase_active_peak_bytes.conditioning)
        );
        assert_eq!(
            derived.denoise,
            expect(anchor.phase_active_peak_bytes.denoise)
        );
        assert_eq!(
            derived.decode,
            expect(anchor.phase_active_peak_bytes.decode)
        );
    }

    #[test]
    fn derived_peaks_grow_with_frames_and_area_in_the_plain_regime() {
        let anchor = store()
            .anchor_for("ltx_2_5", AnchorBackend::Mlx, "q4")
            .expect("q4 anchor");
        let small = anchor
            .derive_video_phase_peaks(plain_request(768, 512, 89))
            .expect("small derivable");
        let longer = anchor
            .derive_video_phase_peaks(plain_request(768, 512, 145))
            .expect("longer derivable");
        let larger = anchor
            .derive_video_phase_peaks(plain_request(1280, 704, 89))
            .expect("larger derivable");
        assert!(longer.peak_bytes() > small.peak_bytes());
        assert!(larger.peak_bytes() > small.peak_bytes());
    }

    #[test]
    fn degenerate_geometry_is_not_derivable() {
        let anchor = store()
            .anchor_for("ltx_2_5", AnchorBackend::Mlx, "q4")
            .expect("q4 anchor");
        assert!(anchor
            .derive_video_phase_peaks(plain_request(0, 512, 89))
            .is_none());
        assert!(anchor
            .derive_video_phase_peaks(plain_request(768, 0, 89))
            .is_none());
        assert!(anchor
            .derive_video_phase_peaks(plain_request(768, 512, 0))
            .is_none());
    }

    // -------------------------------------------------------------------------------------
    // AC 2: the derivation reproduces every retained LTX-2.5 MLX measured peak within the
    // per-term justified margins (epic acceptance test 2, v1).
    // -------------------------------------------------------------------------------------

    #[test]
    fn derivation_brackets_every_retained_corpus_peak() {
        let corpus = retained_corpus();
        // Shape, not a frozen count: every anchored tier must be exercised by the corpus, and
        // both bounded regimes must appear, or this test silently stops asking its question.
        for tier in ["q4", "q8", "bf16"] {
            assert!(
                corpus.iter().any(|record| record.tier == tier),
                "the retained corpus must cover tier {tier}"
            );
        }
        assert!(corpus.iter().any(|record| record.decode_tiled));
        assert!(corpus.iter().any(|record| record.transformer_windowed));
        assert!(corpus.iter().any(|record| record.deferred));

        for record in corpus {
            let anchor = store()
                .anchor_for("ltx_2_5", AnchorBackend::Mlx, &record.tier)
                .unwrap_or_else(|| panic!("tier {} carries an anchor", record.tier));
            let derived = anchor
                .derive_video_phase_peaks(AnchorDeriveRequest {
                    width: record.width,
                    height: record.height,
                    frames: record.frames,
                    decode_tiled: record.decode_tiled,
                    transformer_windowed: record.transformer_windowed,
                    deferred_materialization: record.deferred,
                })
                .unwrap_or_else(|| panic!("record {} is derivable", record.id));

            // Per-phase safety: the margin-widened derived phase covers the measured active peak.
            for (phase, derived_bytes, measured_bytes) in [
                (
                    "conditioning",
                    derived.conditioning,
                    record.conditioning_active,
                ),
                ("denoise", derived.denoise, record.denoise_active),
                ("decode", derived.decode, record.decode_active),
            ] {
                assert!(
                    derived_bytes >= measured_bytes,
                    "record {} phase {phase}: derived {derived_bytes} under measured \
                     {measured_bytes}",
                    record.id
                );
            }

            // Overall bracket: at or above the measured allocator envelope, and within the
            // validation tightness budget so the margins stay falsifiable.
            let upper = derived.peak_bytes();
            assert!(
                upper >= record.overall_envelope,
                "record {}: derived admission bound {upper} under the measured allocator \
                 envelope {}",
                record.id,
                record.overall_envelope
            );
            let tight_cap = ((record.overall_envelope as f64)
                * (1.0 + ANCHOR_VALIDATION_TIGHTNESS_BUDGET))
                .ceil() as u64;
            assert!(
                upper <= tight_cap,
                "record {}: derived admission bound {upper} exceeds the tightness cap \
                 {tight_cap} over envelope {}",
                record.id,
                record.overall_envelope
            );
        }
    }
}
