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
//!
//! # Catalog coverage (sc-22510)
//!
//! The store spans the WHOLE routing catalog, image and video, both backend lanes, and it has
//! exactly two kinds of row: a measured [`MemoryAnchor`], or an explicit [`AnalyticOnlyEntry`] for
//! a `(model, tier, backend lane)` the retained evidence cannot anchor. There is no third state —
//! an unclassified cell is a defect, not a silence — and a cell is classified exactly once.
//! `scripts/extract-memory-anchors.mjs` writes both halves from the retained corpora; its test
//! holds the catalog-coverage and determinism assertions, which need the routing catalog this
//! crate cannot see.
//!
//! Because the store is no longer one corpus, the pipeline axes and the output rate are OPTIONAL:
//! an image capture states no transformer variant, decoder or fps. Both are bound to the source
//! record in BOTH directions, and an axis-free anchor answers no variant-keyed lookup.
//!
//! # Validated domain
//!
//! An anchor's identity cell is `(model, tier, backend lane, transformer variant, decoder)` — the
//! same pipeline axes the fitted curves key on. A request whose variant/decoder differ from the
//! anchor's is NOT derivable: the retained corpus contains no dev-vs-distilled pair at a common
//! regime, so the variant effect is unmeasurable from this evidence and must not be assumed away.
//!
//! An anchor's own measured REGIME is equally load-bearing. A phase that reuses the anchor's
//! measured intercept requires the anchor to have been measured in that phase's unbounded regime;
//! a phase priced by an architecture bound ([`COND_DEFERRED_BOUND_BYTES`],
//! [`DENOISE_WINDOWED_BASE_BYTES`], [`decode_tiled_bound_bytes`]) does not touch the anchor at all.
//! `derive_video_phase_peaks` enforces exactly that and returns `None` (fail open to the caller's
//! floor) otherwise — a deferred-measured conditioning intercept applied to an eager request would
//! under-estimate by the entire transformer residency.
//!
//! The retained LTX-2.5 MLX corpus measures frames at 145 and 449 only (fps 24 and 30). The frames
//! term is therefore validated downward from f449 to f145 on `bf16 dev/diffvae` and nowhere else;
//! every other cell's independent evidence varies AREA at f145. Frame extrapolation above f449 is
//! architecture-justified (latent frames are affine in frames), not corpus-validated.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::memory_calibration::{Ltx25Decoder, Ltx25TransformerVariant};

pub const MEMORY_ANCHOR_SCHEMA_VERSION: u32 = 1;

/// The checked-in anchor store.
pub const PACKAGED_MEMORY_ANCHORS: &str = include_str!("../../../config/memory-anchors.json");

/// Immutable retained evidence the anchors were extracted from. Every anchor's `source.path` must
/// name one of these files, whose bytes are compiled in so the handshake cannot be bypassed by
/// editing the file on disk.
const PACKAGED_MEMORY_ANCHOR_SOURCES: &[(&str, &str)] = &[
    (
        "docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json",
        include_str!("../../../docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json"),
    ),
    (
        "docs/generated/ltx-mlx-geometry-sweep-sc-18810.json",
        include_str!("../../../docs/generated/ltx-mlx-geometry-sweep-sc-18810.json"),
    ),
    (
        "docs/generated/memory-calibration-evidence.json",
        include_str!("../../../docs/generated/memory-calibration-evidence.json"),
    ),
];

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
// Derivation coefficients.
//
// E3 requires per-term-justified margins, so the two kinds of uncertainty are kept apart:
//
// * COEFFICIENT uncertainty lives INSIDE each coefficient. Every slope below is set at or above
//   the highest slope the retained corpus measures WITHIN one identity cell and regime (pairs that
//   differ only in geometry), so no coefficient sits mid-spread and none of them leans on the
//   margin to stay above the trend.
// * The MLX ALLOCATOR envelope above a phase's ACTIVE peak is the one remaining unmodelled term,
//   and it is the only thing [`ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`] is asked to cover.
//
// Every bound constant also names the TIER it was measured on and why that tier upper-bounds the
// others. The corpus validation test (`derivation_brackets_every_retained_corpus_peak`) is the
// falsifier for all of it.
// ---------------------------------------------------------------------------------------------

/// Conditioning-phase bytes per latent token: the packed video latent plus the Gemma text
/// cross-attention context held in fp32 workspace (4096 f32 lanes x 8 concurrently live per-token
/// tensors). Within-cell retained slopes: 109,944 B/token (`q8 dev/diffvae`) and 130,884 B/token
/// (`bf16 dev/diffvae`). Set at 128 KiB/token, above the highest measured slope.
pub const COND_PER_TOKEN_BYTES: i128 = 131_072;

/// Conditioning-phase bound under deferred materialization: only the projected text embeddings and
/// latent init are resident (the transformer stays unmaterialized), which the retained captures
/// show as geometry-independent (11.73 GB active / 12.02 GB allocator at 57.0M and 130.7M output
/// voxels alike). Bound set above the allocator figure.
///
/// TIER PROVENANCE: measured on `bf16 distilled` captures only. It upper-bounds q8/q4 because in
/// this regime the transformer weights are not resident at all — what is held is the projected text
/// embedding and latent-init workspace in fp32, which weight quantization does not shrink.
pub const COND_DEFERRED_BOUND_BYTES: u64 = 12_900_000_000;

/// Denoise-phase bytes per latent token: the DiT forward's live activation set (residual stream,
/// attention chunk workspace under the declared `attentionChunkSize`, MLP intermediates) at fp32.
/// Within-cell retained slopes: 306,560 (`q8 dev/diffvae`), 328,805 and 329,431 B/token
/// (`bf16 distilled/diffvae`, `bf16 dev/diffvae`). Set at 328 KiB/token, above the highest.
pub const DENOISE_PER_TOKEN_BYTES: i128 = 335_872;

/// Denoise-phase intercept under `bounded_transformer_residency` (window size 1): one resident
/// AvDiT block of the declared 48 plus the bounded attention workspace, replacing the full
/// transformer residency an unwindowed anchor was measured with. The per-token activation slope is
/// unchanged by windowing ([`DENOISE_PER_TOKEN_BYTES`]).
///
/// TIER PROVENANCE: measured on `bf16 distilled` windowed captures only (implied intercept 2.65 GB
/// at both measured token counts). bf16 is the WIDEST tier — one resident AvDiT block is strictly
/// larger at bf16 than at q8/q4 — so this upper-bounds the quantized tiers.
pub const DENOISE_WINDOWED_BASE_BYTES: u64 = 2_650_000_000;

/// Decode-phase bytes per output voxel in the single-pass DIFFVAE regime: concurrently live
/// pixel-space working copies. Every non-tiled decode record in the retained corpus is a diffvae
/// record — single-pass conv decode is UNMEASURED, and the identity/regime guards in
/// [`MemoryAnchor::derive_video_phase_peaks`] keep it out of this law rather than pricing it here.
///
/// Within-cell retained slopes: 30.57 (`q8 dev/diffvae`), 48.92 (`bf16 dev/diffvae`), 62.81 and
/// 62.87 B/voxel (`bf16 distilled/diffvae`). Set at 63 B/voxel — the observed UPPER slope — so the
/// bound cannot cross below the measured trend at large output geometries.
pub const DECODE_PER_VOXEL_BYTES: i128 = 63;

/// Tile-bounded decode WORKSPACE under `bounded_decode` (declared `decodeTileEdge`/`decodeOverlap`):
/// decoder weights plus the tile-sized activation working set, which the declared tile parameters
/// make geometry-independent by construction of the tiling.
///
/// Measured on the two retained tiled captures — `bf16 distilled/conv` 8.23 GB and `q8 dev/conv`
/// 8.26 GB. Both sit at the SAME 130.7M output voxels, so the geometry-independence of this term
/// rests on the tiling contract, not on two measured points; subtracting
/// [`DECODE_TILED_PER_VOXEL_BYTES`] at that geometry leaves an implied workspace of 6.66-6.69 GB
/// and this constant is set above it. bf16 is the widest tier for the decoder weights, so it
/// upper-bounds q8/q4.
pub const DECODE_TILE_WORKSPACE_BYTES: i128 = 7_000_000_000;

/// Tiling bounds the decode WORKSPACE, not the OUTPUT: a tiled decode still materializes the whole
/// output clip, so the tiled decode estimate cannot be a flat constant. This term is
/// architecture-determined rather than fitted — `width x height x frames x 3 RGB channels x 4 bytes`
/// of fp32 pixel space, the widest element the decode path materializes before delivery.
pub const DECODE_TILED_PER_VOXEL_BYTES: i128 = 12;

/// Tile-bounded decode estimate at one output geometry: the geometry-independent workspace plus the
/// output clip the tiling does not bound.
pub const fn decode_tiled_bound_bytes(voxels: i128) -> i128 {
    DECODE_TILE_WORKSPACE_BYTES + DECODE_TILED_PER_VOXEL_BYTES * voxels
}

/// Multiplicative margin applied to every derived phase peak. It covers exactly ONE term: the MLX
/// allocator envelope that sits above a phase's ACTIVE peak (cache retention across phase
/// transitions), observed at up to 15.84% over the binding active phase across the retained corpus.
/// Coefficient uncertainty is NOT covered here — it is priced inside the coefficients above, each
/// of which is set at or above the highest measured within-cell slope.
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
    /// Catalog cells the retained evidence cannot anchor (sc-22510). This is the store's SECOND
    /// entry kind, not a comment: a `(model, tier, backend lane)` the corpus never measured with a
    /// per-phase decomposition is derived from architecture facts alone, and saying so explicitly
    /// is what makes "unmeasured" distinguishable from "missing row". `default` so a store written
    /// before the migration still parses.
    #[serde(default)]
    pub analytic_only: Vec<AnalyticOnlyEntry>,
}

/// Why a cell is analytic-only, strongest evidence first. The variant names the BEST evidence that
/// exists for the cell — never a phase decomposition, or it would be an anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalyticOnlyBasis {
    /// A retained render measures the cell's overall allocator envelope but no per-phase split.
    MeasuredEnvelope,
    /// The pinned MLX provider publishes measured component/stage byte constants.
    ProviderMeasuredConstants,
    /// The catalog manifest declares a `measured: true` per-tier envelope.
    ManifestTierDeclaration,
    /// Nothing measured covers the cell at all.
    NoRetainedEvidence,
}

/// Provenance for an analytic-only classification.
///
/// Unlike [`AnchorSource`] this is NOT a byte-exact handshake against compiled-in evidence: an
/// analytic-only row feeds no derivation, so it carries provenance a reader can follow rather than
/// a digest the estimate depends on. Evidence living outside this repo (the pinned inference
/// revision's provider constants) is named by `repo` + `revision` precisely because it cannot be
/// compiled in and re-hashed here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnalyticOnlyEvidence {
    /// `None` for this repository; otherwise the foreign repo the path is relative to.
    pub repo: Option<String>,
    /// The revision the foreign path was read at — required whenever `repo` is set, so a foreign
    /// citation can never mean "whatever that repo holds now".
    pub revision: Option<String>,
    pub path: String,
    pub sha256: String,
    pub record_id: Option<String>,
    pub envelope_bytes: Option<u64>,
    /// Named scalars carried verbatim as text (declared GB, provider byte constants), so a value
    /// cannot change meaning through float re-formatting between the generator and this parser.
    pub values: Option<BTreeMap<String, String>>,
}

/// One catalog cell that is derived analytically rather than from a measured anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnalyticOnlyEntry {
    pub id: String,
    pub model_id: String,
    pub model_family: String,
    pub route: String,
    pub backend: AnchorBackend,
    pub tier: String,
    pub basis: AnalyticOnlyBasis,
    /// Human-readable statement of what is missing. Empty is refused: a classification with no
    /// stated reason is a gap wearing a row's clothes.
    pub reason: String,
    pub evidence: Option<AnalyticOnlyEvidence>,
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
    /// Output rate, bound to the record's `outputFps` measurement. `None` where the record states
    /// none — every image capture, whose single frame has no rate — and the absence is checked in
    /// BOTH directions so a missing rate cannot be silently invented or dropped.
    pub fps: Option<u32>,
}

/// The measured per-phase ACTIVE peak decomposition of the anchor render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnchorPhaseBytes {
    pub conditioning: u64,
    pub denoise: u64,
    pub decode: u64,
}

/// The bounded rungs the anchor render itself engaged, taken from the source record's
/// `strategy.engagedRungs`. A phase whose derivation reuses the anchor's measured intercept
/// requires the anchor to have run that phase UNBOUNDED; see
/// [`MemoryAnchor::derive_video_phase_peaks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnchorMeasuredRegime {
    /// `bounded_decode` was engaged for the anchor render.
    pub decode_tiled: bool,
    /// `bounded_transformer_residency` was engaged for the anchor render.
    pub transformer_windowed: bool,
}

/// One measured anchor: identity plus the peak decomposition of exactly one retained render.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryAnchor {
    pub id: String,
    pub model_id: String,
    pub model_family: String,
    /// Resolved engine id. The retained LTX-2.5 corpus carries no `target.route`, so this binds
    /// against `target.provider` exactly as `video_memory_curves::source_route` does.
    pub route: String,
    /// Provider descriptor id the anchor render was measured against. The measurement pins one
    /// artifact repository/revision, so a different provider answering for the same catalog model
    /// must not inherit it.
    pub provider: String,
    pub backend: AnchorBackend,
    /// Plan tier key: `q4` / `q8` / `bf16` / ...
    pub tier: String,
    /// LTX-2.5 pipeline identity — the same axes the fitted curves key on. The corpus has no
    /// dev-vs-distilled pair at a common regime, so a request on the other variant/decoder is not
    /// derivable from this anchor.
    ///
    /// `None` where the source record states no pipeline axes at all (sc-22510): every image
    /// capture, and the LTX-2.3 sweep. A `None` anchor answers no variant-keyed lookup — it is
    /// not a wildcard — because a record that never named a variant cannot certify one.
    pub transformer_variant: Option<Ltx25TransformerVariant>,
    pub decoder: Option<Ltx25Decoder>,
    pub mode: String,
    /// The anchor was measured overlay-free; a future overlay anchor is a new row, never a reuse.
    pub overlay: Option<String>,
    pub reference_count: u32,
    /// Materialization shape the anchor render ran under. This is LOAD-BEARING, not informational:
    /// a deferred-measured conditioning intercept does not price an eager request (the transformer
    /// residency is missing from it entirely), so the derivation refuses that combination.
    pub load_shape: AnchorLoadShape,
    /// The bounded rungs the anchor render itself engaged, for the same reason.
    pub measured_regime: AnchorMeasuredRegime,
    pub source: AnchorSource,
    pub geometry: AnchorGeometry,
    pub phase_active_peak_bytes: AnchorPhaseBytes,
    /// The measured overall allocator envelope of the anchor render (active + reclaimable).
    pub overall_allocator_envelope_bytes: u64,
}

/// Store-key spelling of the LTX-2.5 transformer variant, matching the retained evidence.
pub const fn transformer_variant_key(variant: Ltx25TransformerVariant) -> &'static str {
    match variant {
        Ltx25TransformerVariant::Distilled => "distilled",
        Ltx25TransformerVariant::Dev => "dev",
    }
}

/// Store-key spelling of the LTX-2.5 decoder, matching the retained evidence.
pub const fn decoder_key(decoder: Ltx25Decoder) -> &'static str {
    match decoder {
        Ltx25Decoder::Conv => "conv",
        Ltx25Decoder::DiffVae => "diffvae",
    }
}

/// Key spelling of an optional pipeline axis. A record that states no variant/decoder keys on
/// `"-"`, which no stated axis spells, so the two can never collide in the identity map.
fn optional_axis_key(value: Option<&str>) -> &str {
    value.unwrap_or("-")
}

fn variant_key_opt(variant: Option<Ltx25TransformerVariant>) -> &'static str {
    match variant {
        Some(variant) => transformer_variant_key(variant),
        None => "-",
    }
}

fn decoder_key_opt(decoder: Option<Ltx25Decoder>) -> &'static str {
    match decoder {
        Some(decoder) => decoder_key(decoder),
        None => "-",
    }
}

/// The `(model, tier, backend lane)` coverage cell — the unit the routing catalog enumerates and
/// the unit every cell must be classified in exactly once, as either an anchor or analytic-only.
fn coverage_cell(
    model_id: &str,
    tier: &str,
    backend: AnchorBackend,
) -> (String, String, AnchorBackend) {
    (model_id.to_owned(), tier.to_owned(), backend)
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
        let identity = (
            anchor.model_id.clone(),
            anchor.tier.clone(),
            anchor.backend,
            variant_key_opt(anchor.transformer_variant),
            decoder_key_opt(anchor.decoder),
        );
        if let Some(previous) = identities.insert(identity, &anchor.id) {
            return Err(format!(
                "duplicate memory anchor for ({}, {}, {}, {}, {}): {} and {} — exactly one anchor \
                 per (model, tier, backend lane, transformer variant, decoder) is the schema \
                 invariant",
                anchor.model_id,
                anchor.tier,
                anchor.backend.as_key(),
                variant_key_opt(anchor.transformer_variant),
                decoder_key_opt(anchor.decoder),
                previous,
                anchor.id
            ));
        }
        validate_anchor(anchor)?;
    }
    validate_analytic_only(&store)?;
    Ok(store)
}

/// The analytic-only half of the store's invariants (sc-22510).
///
/// A cell is classified exactly once. An anchored cell may not ALSO be declared analytic-only —
/// that would let a measured cell be quietly demoted to an architecture estimate while the anchor
/// sat unused — and two analytic rows for one cell would let a reader draw either conclusion.
fn validate_analytic_only(store: &MemoryAnchorStore) -> Result<(), String> {
    let anchored: std::collections::BTreeSet<_> = store
        .anchors
        .iter()
        .map(|anchor| coverage_cell(&anchor.model_id, &anchor.tier, anchor.backend))
        .collect();
    let mut seen = BTreeMap::new();
    for entry in &store.analytic_only {
        if entry.reason.trim().is_empty() {
            return Err(format!(
                "analytic-only entry {} states no reason — an unexplained classification is a gap",
                entry.id
            ));
        }
        let cell = coverage_cell(&entry.model_id, &entry.tier, entry.backend);
        if anchored.contains(&cell) {
            return Err(format!(
                "analytic-only entry {} names ({}, {}, {}), which carries a measured anchor — a \
                 cell is classified exactly once",
                entry.id,
                entry.model_id,
                entry.tier,
                entry.backend.as_key()
            ));
        }
        if let Some(previous) = seen.insert(cell, &entry.id) {
            return Err(format!(
                "duplicate analytic-only entry for ({}, {}, {}): {} and {}",
                entry.model_id,
                entry.tier,
                entry.backend.as_key(),
                previous,
                entry.id
            ));
        }
        match (entry.basis, entry.evidence.as_ref()) {
            (AnalyticOnlyBasis::NoRetainedEvidence, Some(_)) => {
                return Err(format!(
                    "analytic-only entry {} claims no retained evidence yet cites some",
                    entry.id
                ));
            }
            (AnalyticOnlyBasis::NoRetainedEvidence, None) => {}
            (_, None) => {
                return Err(format!(
                    "analytic-only entry {} names an evidence basis but cites no evidence",
                    entry.id
                ));
            }
            (_, Some(evidence)) => validate_analytic_evidence(&entry.id, evidence)?,
        }
    }
    Ok(())
}

fn validate_analytic_evidence(id: &str, evidence: &AnalyticOnlyEvidence) -> Result<(), String> {
    if evidence.sha256.len() != 64 || !evidence.sha256.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "analytic-only entry {id} cites a malformed source digest"
        ));
    }
    if evidence.repo.is_some() && evidence.revision.is_none() {
        return Err(format!(
            "analytic-only entry {id} cites a foreign repository without the revision it was read \
             at — a citation that floats with someone else's default branch proves nothing"
        ));
    }
    // A repo-local citation of a file that happens to be compiled in gets the same byte-exact
    // handshake the anchors get; one that is not compiled in carries provenance only, which is all
    // an analytic-only row is asked for (it feeds no derivation).
    if evidence.repo.is_none() {
        let path = evidence
            .path
            .split_once('#')
            .map_or(evidence.path.as_str(), |(file, _)| file);
        if let Some((_, raw)) = PACKAGED_MEMORY_ANCHOR_SOURCES
            .iter()
            .find(|(candidate, _)| *candidate == path)
        {
            let digest = format!("{:x}", Sha256::digest(raw.as_bytes()));
            if digest != evidence.sha256 {
                return Err(format!(
                    "analytic-only entry {id} source digest mismatch for {path}: recorded {} \
                     actual {digest}",
                    evidence.sha256
                ));
            }
        }
    }
    Ok(())
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
    // `target.route` is absent from the LTX-2.5 seed; `video_memory_curves::source_route` treats
    // `target.provider` as the route spelling in exactly that case, and so does this.
    let record_route = str_at(target, "route").or_else(|| str_at(target, "provider"));
    if str_at(target, "modelId").as_deref() != Some(anchor.model_id.as_str())
        || str_at(target, "tier").as_deref() != Some(anchor.tier.as_str())
        || str_at(record, "backend").as_deref() != Some(anchor.backend.as_key())
        || str_at(target, "mode").as_deref() != Some(anchor.mode.as_str())
        || str_at(target, "provider").as_deref() != Some(anchor.provider.as_str())
        || record_route.as_deref() != Some(anchor.route.as_str())
        // Both directions: a record that states a pipeline axis must have it recorded, and one
        // that states none must not acquire one in the store (sc-22510).
        || optional_axis_key(str_at(target, "transformerVariant").as_deref())
            != variant_key_opt(anchor.transformer_variant)
        || optional_axis_key(str_at(target, "decoder").as_deref())
            != decoder_key_opt(anchor.decoder)
        || str_at(record, "calibrationFingerprint").as_deref()
            != Some(anchor.source.calibration_fingerprint.as_str())
    {
        return Err(format!(
            "memory anchor {} identity disagrees with its source record {}",
            anchor.id, anchor.source.record_id
        ));
    }
    let load_shape_key = match anchor.load_shape {
        AnchorLoadShape::EagerMaterialization => "eager_materialization",
        AnchorLoadShape::DeferredMaterialization => "deferred_materialization",
    };
    if str_at(record, "loadShape").as_deref() != Some(load_shape_key) {
        return Err(format!(
            "memory anchor {} load shape disagrees with its source record {}",
            anchor.id, anchor.source.record_id
        ));
    }
    let engaged: Vec<&str> = record
        .get("strategy")
        .and_then(|strategy| strategy.get("engagedRungs"))
        .and_then(|rungs| rungs.as_array())
        .map(|rungs| rungs.iter().filter_map(|rung| rung.as_str()).collect())
        .unwrap_or_default();
    if anchor.measured_regime.decode_tiled != engaged.contains(&"bounded_decode")
        || anchor.measured_regime.transformer_windowed
            != engaged.contains(&"bounded_transformer_residency")
    {
        return Err(format!(
            "memory anchor {} measured regime disagrees with the engaged rungs of its source \
             record {}",
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
    // Output rate is evidence identity, not decoration: bind it to the measured `outputFps` rather
    // than leaving a serialized field nothing checks.
    if measured.get("outputFps").copied() != anchor.geometry.fps.map(u64::from) {
        return Err(format!(
            "memory anchor {} fps {:?} disagrees with the outputFps measurement of its source \
             record {}",
            anchor.id, anchor.geometry.fps, anchor.source.record_id
        ));
    }
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
    /// The unique anchor for one `(model, backend lane, tier, transformer variant, decoder)`
    /// coordinate. The pipeline axes are part of the key, not a post-filter: the corpus measures no
    /// dev-vs-distilled pair at a common regime, so one variant's anchor may not price the other's
    /// render.
    pub fn anchor_for(
        &self,
        model_id: &str,
        backend: AnchorBackend,
        tier: &str,
        transformer_variant: Ltx25TransformerVariant,
        decoder: Ltx25Decoder,
    ) -> Option<&MemoryAnchor> {
        self.anchors.iter().find(|anchor| {
            anchor.model_id == model_id
                && anchor.backend == backend
                && anchor.tier == tier
                // An anchor whose record stated no pipeline axes (`None`) answers no variant-keyed
                // lookup: it certifies neither variant, so it must not stand in for either.
                && anchor.transformer_variant == Some(transformer_variant)
                && anchor.decoder == Some(decoder)
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
    /// `bounded_decode` engaged: decode is tile-bounded ([`decode_tiled_bound_bytes`]).
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
    let latent_frames = 1 + u64::from(frames)
        .saturating_sub(1)
        .div_ceil(LTX_TEMPORAL_SCALE);
    i128::from(patches_w) * i128::from(patches_h) * i128::from(latent_frames)
}

fn voxels(width: u32, height: u32, frames: u32) -> i128 {
    i128::from(width) * i128::from(height) * i128::from(frames)
}

/// Widen one derived phase estimate by [`ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`] in integer bytes.
///
/// A non-positive pre-margin value is NOT clamped to zero: an extrapolation that ran below the
/// anchor far enough to go negative has left the law's domain, and a 0-byte peak would admit
/// anything. `None` fails open to the caller's floor instead.
fn widened(bytes: i128) -> Option<u64> {
    if bytes <= 0 {
        return None;
    }
    let bytes = u64::try_from(bytes).ok()?;
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
    ///   tile-bounded estimate ([`decode_tiled_bound_bytes`]).
    ///
    /// REGIME GUARD: a phase priced from the anchor's own measured intercept requires the anchor to
    /// have been measured in that phase's UNBOUNDED regime. A deferred-materialization anchor
    /// carries no transformer residency in its conditioning peak; a windowed anchor carries one
    /// AvDiT block instead of 48 in its denoise peak; a tiled anchor carries a tile workspace
    /// instead of the full decode working set. Reusing any of those for an unbounded request would
    /// under-estimate by the whole omitted residency, so the derivation refuses instead.
    ///
    /// Returns `None` on degenerate geometry, on a regime the anchor cannot price, and on any
    /// extrapolation that runs non-positive; every estimate is widened by
    /// [`ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`].
    pub fn derive_video_phase_peaks(
        &self,
        request: AnchorDeriveRequest,
    ) -> Option<AnchorDerivedPhases> {
        if request.width == 0 || request.height == 0 || request.frames == 0 {
            return None;
        }
        let anchor_deferred = self.load_shape == AnchorLoadShape::DeferredMaterialization;
        if (!request.deferred_materialization && anchor_deferred)
            || (!request.transformer_windowed && self.measured_regime.transformer_windowed)
            || (!request.decode_tiled && self.measured_regime.decode_tiled)
        {
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
            decode_tiled_bound_bytes(voxels)
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
        transformer_variant: Ltx25TransformerVariant,
        decoder: Ltx25Decoder,
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
                    transformer_variant: match record["target"]["transformerVariant"].as_str() {
                        Some("dev") => Ltx25TransformerVariant::Dev,
                        Some("distilled") => Ltx25TransformerVariant::Distilled,
                        other => panic!("unknown transformer variant {other:?}"),
                    },
                    decoder: match record["target"]["decoder"].as_str() {
                        Some("conv") => Ltx25Decoder::Conv,
                        Some("diffvae") => Ltx25Decoder::DiffVae,
                        other => panic!("unknown decoder {other:?}"),
                    },
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

    /// Index of an LTX-2.5 anchor in the packaged store. Since sc-22510 the store spans the whole
    /// catalog, so the doctoring tests below cannot assume `anchors[0]` is a video row.
    fn ltx25_anchor_index() -> usize {
        store()
            .anchors
            .iter()
            .position(|anchor| {
                anchor.model_id == "ltx_2_5" && anchor.source.path == LTX25_CORPUS_PATH
            })
            .expect("the store carries an LTX-2.5 anchor")
    }

    // -------------------------------------------------------------------------------------
    // AC 1 (store half): every LTX-2.5 MLX (tier, transformer variant, decoder) cell the
    // retained corpus measures carries exactly one anchor, and no cell it does not measure
    // carries any.
    // -------------------------------------------------------------------------------------

    #[test]
    fn ltx25_mlx_carries_exactly_one_anchor_per_measured_pipeline_cell() {
        let corpus = retained_corpus();
        let measured: std::collections::BTreeSet<(String, &str, &str)> = corpus
            .iter()
            .map(|record| {
                (
                    record.tier.clone(),
                    transformer_variant_key(record.transformer_variant),
                    decoder_key(record.decoder),
                )
            })
            .collect();
        // Shape, not a frozen count: the corpus spans more than one variant and more than one
        // decoder, so this test cannot silently degenerate into a single-cell assertion.
        assert!(
            measured
                .iter()
                .map(|(_, variant, _)| *variant)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                > 1,
            "the retained corpus must span more than one transformer variant"
        );
        assert!(
            measured
                .iter()
                .map(|(_, _, decoder)| *decoder)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                > 1,
            "the retained corpus must span more than one decoder"
        );

        for (tier, variant_key, decoder_key_str) in &measured {
            let matching = store()
                .anchors
                .iter()
                .filter(|anchor| {
                    anchor.model_id == "ltx_2_5"
                        && anchor.backend == AnchorBackend::Mlx
                        && anchor.tier == *tier
                        && variant_key_opt(anchor.transformer_variant) == *variant_key
                        && decoder_key_opt(anchor.decoder) == *decoder_key_str
                })
                .count();
            assert_eq!(
                matching, 1,
                "cell ({tier}, {variant_key}, {decoder_key_str}) must carry exactly one anchor"
            );
        }
        assert_eq!(
            store()
                .anchors
                .iter()
                .filter(
                    |anchor| anchor.model_id == "ltx_2_5" && anchor.backend == AnchorBackend::Mlx
                )
                .count(),
            measured.len(),
            "the store must not carry an anchor for a pipeline cell the corpus never measured"
        );
        // Scoped to LTX-2.5: since sc-22510 the store spans the whole routing catalog, so this
        // asks that the LTX-2.5 rows still come from the LTX-2.5 corpus and nothing more.
        for anchor in store()
            .anchors
            .iter()
            .filter(|anchor| anchor.model_id == "ltx_2_5")
        {
            assert_eq!(anchor.source.path, LTX25_CORPUS_PATH);
        }
    }

    // -------------------------------------------------------------------------------------
    // sc-22510: the store is catalog-wide, and every row is one of exactly two kinds.
    // -------------------------------------------------------------------------------------

    /// The store's shape, asserted without pinning any population.
    ///
    /// sc-22512: this deliberately does NOT require the store to declare any analytic-only row. A
    /// cell nobody measured and nobody classified is priced from the conservative analytic estimate
    /// at admission time; its absence from this file is never a build failure. What IS asserted is
    /// the shape of the rows that ARE present, which holds at any store size including zero.
    #[test]
    fn every_store_row_is_well_formed_whatever_the_store_holds() {
        // sc-22512 removed the three population floors that stood here: "the store must keep at
        // least one pipeline-keyed (video) anchor", "must carry anchors extracted from corpora that
        // state no pipeline axes", and "the migration covers more than one model". Each reddened for
        // exactly one reason — a MEASUREMENT not being in the store — so retiring a corpus, or
        // shipping a catalog nobody has captured yet, broke the build over bookkeeping. Under E8 an
        // unanchored cell is priced from the conservative analytic estimate at admission time; its
        // absence here is never a failure.
        //
        // Every claim below is universally quantified, so it keeps full force over whatever rows the
        // store holds and degrades to vacuous rather than to red when it holds none.
        for anchor in &store().anchors {
            assert!(
                !anchor.model_id.trim().is_empty() && !anchor.tier.trim().is_empty(),
                "an anchor must name the cell it prices"
            );
            // A pipeline-keyed anchor states BOTH axes or NEITHER. Half a key would answer a
            // variant-keyed lookup it was never measured under.
            assert_eq!(
                anchor.transformer_variant.is_some(),
                anchor.decoder.is_some(),
                "{}: an anchor's pipeline axes are stated together or not at all",
                anchor.id
            );
        }
        // Shape, not population: every analytic-only row that IS present states a reason and does
        // not collide with an anchored cell. Zero such rows is a legal store.
        for entry in &store().analytic_only {
            assert!(
                !entry.reason.trim().is_empty(),
                "{}: an analytic-only classification must state its reason",
                entry.id
            );
            assert!(
                !store().anchors.iter().any(|anchor| {
                    anchor.model_id == entry.model_id
                        && anchor.backend == entry.backend
                        && anchor.tier == entry.tier
                }),
                "{}: a cell is classified exactly once — it is anchored and analytic-only",
                entry.id
            );
        }
    }

    /// An anchor whose source record stated NO pipeline axes must not answer a variant-keyed
    /// lookup: it certifies neither variant, and standing in for one would price a request from a
    /// render that never claimed to be it.
    #[test]
    fn an_axis_free_anchor_answers_no_variant_keyed_lookup() {
        let axis_free = store()
            .anchors
            .iter()
            .find(|anchor| anchor.transformer_variant.is_none())
            .expect("an axis-free anchor exists");
        for variant in [
            Ltx25TransformerVariant::Dev,
            Ltx25TransformerVariant::Distilled,
        ] {
            for decoder in [Ltx25Decoder::Conv, Ltx25Decoder::DiffVae] {
                assert!(
                    store()
                        .anchor_for(
                            &axis_free.model_id,
                            axis_free.backend,
                            &axis_free.tier,
                            variant,
                            decoder
                        )
                        .is_none(),
                    "an axis-free anchor must not resolve for ({variant:?}, {decoder:?})"
                );
            }
        }
    }

    /// A cell is classified exactly once. Declaring an anchored cell analytic-only would demote a
    /// measured cell to an architecture estimate with the anchor sitting unused beside it.
    #[test]
    fn an_analytic_entry_for_an_anchored_cell_is_rejected() {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        let anchor = doctored["anchors"][0].clone();
        doctored["analyticOnly"]
            .as_array_mut()
            .expect("analytic-only rows")
            .push(serde_json::json!({
                "id": "analytic:collision",
                "modelId": anchor["modelId"],
                "modelFamily": anchor["modelFamily"],
                "route": anchor["route"],
                "backend": anchor["backend"],
                "tier": anchor["tier"],
                "basis": "no_retained_evidence",
                "reason": "collides with a measured anchor",
                "evidence": serde_json::Value::Null,
            }));
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("an anchored cell may not also be analytic-only");
        assert!(error.contains("classified exactly once"), "{error}");
    }

    #[test]
    fn a_duplicate_or_unexplained_analytic_entry_is_rejected() {
        // sc-22512: this adversarial harness doctors a REAL row, so it can only ask its question
        // when the corpus supplies one. It used to index `analyticOnly[0]` unconditionally and
        // panicked on an empty list — a store with nothing classified reddened the suite, which is
        // the measurement-absence failure this story removes. Skipping is the E8 posture: absence
        // withholds the question, it never answers it with a failure. The loader rules themselves
        // are unchanged and keep full force on every row the store does carry.
        if store().analytic_only.is_empty() {
            return;
        }
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        let clone = doctored["analyticOnly"][0].clone();
        doctored["analyticOnly"]
            .as_array_mut()
            .expect("analytic-only rows")
            .push(clone);
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("a duplicated analytic cell must be rejected");
        assert!(error.contains("duplicate analytic-only entry"), "{error}");

        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["analyticOnly"][0]["reason"] = serde_json::json!("   ");
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("an unexplained classification must be rejected");
        assert!(error.contains("states no reason"), "{error}");
    }

    /// The basis and the cited evidence must agree in both directions, and a foreign-repo citation
    /// must name the revision it was read at.
    #[test]
    fn an_analytic_basis_that_disagrees_with_its_evidence_is_rejected() {
        // sc-22512: same posture as the harness above. Each leg doctors a row of a specific KIND,
        // so it runs exactly when the corpus carries that kind and is skipped when it does not.
        // The `.expect("an evidenced analytic-only entry exists")` pair that stood here made a
        // corpus with nothing classified — or with only unevidenced classifications — red the
        // suite, which is failure-on-absence rather than on anything malformed.
        let (Some(evidenced), Some(unevidenced)) = (
            store()
                .analytic_only
                .iter()
                .position(|entry| entry.evidence.is_some()),
            store()
                .analytic_only
                .iter()
                .position(|entry| entry.evidence.is_none()),
        ) else {
            return;
        };

        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["analyticOnly"][evidenced]["basis"] = serde_json::json!("no_retained_evidence");
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("claiming no evidence while citing some must be rejected");
        assert!(error.contains("claims no retained evidence"), "{error}");

        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["analyticOnly"][unevidenced]["basis"] = serde_json::json!("measured_envelope");
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("naming a basis with no evidence must be rejected");
        assert!(error.contains("cites no evidence"), "{error}");

        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["analyticOnly"][evidenced]["evidence"]["repo"] =
            serde_json::json!("SceneWorks/inference");
        doctored["analyticOnly"][evidenced]["evidence"]["revision"] = serde_json::Value::Null;
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("a foreign citation without a revision must be rejected");
        assert!(error.contains("without the revision"), "{error}");
    }

    /// A repo-local analytic citation of a COMPILED-IN corpus gets the anchors' byte-exact
    /// handshake: the store may not cite evidence it has drifted from.
    #[test]
    fn an_analytic_citation_of_a_compiled_source_binds_to_its_digest() {
        // The store need not contain such a row: which cells are analytic-only, and what they cite,
        // is a property of the retained corpus, and a measurement leaving it must not red this
        // test. The assertion is about rows that ARE present.
        let Some(index) = store().analytic_only.iter().position(|entry| {
            entry.evidence.as_ref().is_some_and(|evidence| {
                evidence.repo.is_none()
                    && PACKAGED_MEMORY_ANCHOR_SOURCES
                        .iter()
                        .any(|(path, _)| *path == evidence.path)
            })
        }) else {
            return;
        };
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["analyticOnly"][index]["evidence"]["sha256"] = serde_json::json!("0".repeat(64));
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("a drifted analytic citation must be rejected");
        assert!(error.contains("source digest mismatch"), "{error}");
    }

    /// The pipeline cell is part of the lookup key, not a post-filter: the corpus contains no
    /// dev-vs-distilled pair at a common regime, so the variant effect is unmeasurable and one
    /// variant's anchor must never resolve for the other's request.
    #[test]
    fn a_foreign_pipeline_cell_resolves_no_anchor() {
        // Measured at q8: dev/diffvae and dev/conv. Never measured at q8: either distilled combo.
        assert!(store()
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q8",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae
            )
            .is_some());
        assert!(store()
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q8",
                Ltx25TransformerVariant::Distilled,
                Ltx25Decoder::Conv
            )
            .is_none());
        // Measured at q4: dev/diffvae only.
        assert!(store()
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q4",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::Conv
            )
            .is_none());
    }

    /// An anchor measured in a BOUNDED regime carries a truncated intercept for that phase. It may
    /// not price the unbounded request that would reuse it.
    #[test]
    fn an_anchor_measured_in_a_bounded_regime_refuses_the_unbounded_request() {
        let bounded = store()
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "bf16",
                Ltx25TransformerVariant::Distilled,
                Ltx25Decoder::Conv,
            )
            .expect("the bf16 distilled/conv anchor exists");
        assert_eq!(bounded.load_shape, AnchorLoadShape::DeferredMaterialization);
        assert!(bounded.measured_regime.decode_tiled);
        assert!(bounded.measured_regime.transformer_windowed);

        let matching = AnchorDeriveRequest {
            width: bounded.geometry.width,
            height: bounded.geometry.height,
            frames: bounded.geometry.frames,
            decode_tiled: true,
            transformer_windowed: true,
            deferred_materialization: true,
        };
        assert!(bounded.derive_video_phase_peaks(matching).is_some());
        for (label, request) in [
            (
                "eager request against a deferred anchor",
                AnchorDeriveRequest {
                    deferred_materialization: false,
                    ..matching
                },
            ),
            (
                "unwindowed request against a windowed anchor",
                AnchorDeriveRequest {
                    transformer_windowed: false,
                    ..matching
                },
            ),
            (
                "single-pass request against a tiled anchor",
                AnchorDeriveRequest {
                    decode_tiled: false,
                    ..matching
                },
            ),
        ] {
            assert!(
                bounded.derive_video_phase_peaks(request).is_none(),
                "{label} must fall open to the caller's floor"
            );
        }
    }

    /// The store schema may not be doctored into a regime it was not measured in — the handshake
    /// binds `loadShape` and `measuredRegime` to the source record.
    #[test]
    fn a_doctored_load_shape_or_measured_regime_is_rejected() {
        let ltx = ltx25_anchor_index();
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][ltx]["loadShape"] = serde_json::json!("deferred_materialization");
        let error =
            load_memory_anchors(&doctored.to_string()).expect_err("load shape must bind to source");
        assert!(error.contains("load shape disagrees"), "{error}");

        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][ltx]["measuredRegime"]["decodeTiled"] = serde_json::json!(true);
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("measured regime must bind to the engaged rungs");
        assert!(error.contains("measured regime disagrees"), "{error}");
    }

    /// Identity fields that were previously stored-and-unchecked are bound to the source record.
    #[test]
    fn a_doctored_pipeline_identity_provider_route_or_fps_is_rejected() {
        let ltx = ltx25_anchor_index();
        for (field, value) in [
            ("transformerVariant", serde_json::json!("distilled")),
            ("decoder", serde_json::json!("conv")),
            ("provider", serde_json::json!("someone_else")),
            ("route", serde_json::json!("ltx_2_5_other")),
        ] {
            let mut doctored: serde_json::Value =
                serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
            doctored["anchors"][ltx][field] = value;
            let error = load_memory_anchors(&doctored.to_string())
                .err()
                .unwrap_or_else(|| panic!("{field} must bind to the source record"));
            assert!(error.contains("identity disagrees"), "{field}: {error}");
        }

        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        let fps = doctored["anchors"][ltx]["geometry"]["fps"]
            .as_u64()
            .expect("fps");
        doctored["anchors"][ltx]["geometry"]["fps"] = serde_json::json!(fps + 1);
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("fps must bind to the outputFps measurement");
        assert!(error.contains("disagrees with the outputFps"), "{error}");

        // sc-22510 made `fps` and the pipeline axes OPTIONAL, so both are checked in BOTH
        // directions. These two cases exercise the direction a `is_some() && ...` guard would
        // silently drop: a record that DID state a value, dropped to null in the store.
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][ltx]["geometry"]["fps"] = serde_json::Value::Null;
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("dropping a measured output rate must be rejected");
        assert!(error.contains("disagrees with the outputFps"), "{error}");

        for axis in ["transformerVariant", "decoder"] {
            let mut doctored: serde_json::Value =
                serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
            doctored["anchors"][ltx][axis] = serde_json::Value::Null;
            let error = load_memory_anchors(&doctored.to_string())
                .err()
                .unwrap_or_else(|| panic!("dropping {axis} must be rejected"));
            assert!(error.contains("identity disagrees"), "{axis}: {error}");
        }

        // And the other direction: a record that measured NO output rate — every image capture —
        // must not acquire one in the store. Whether such an anchor exists depends on which
        // captures the corpus retains, so this asks its question only when one is there rather than
        // reding when the last rateless capture is retired.
        if let Some(rateless) = store()
            .anchors
            .iter()
            .position(|anchor| anchor.geometry.fps.is_none())
        {
            let mut doctored: serde_json::Value =
                serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
            doctored["anchors"][rateless]["geometry"]["fps"] = serde_json::json!(24);
            let error = load_memory_anchors(&doctored.to_string())
                .expect_err("an invented output rate must be rejected");
            assert!(error.contains("disagrees with the outputFps"), "{error}");
        }

        if let Some(axis_free) = store()
            .anchors
            .iter()
            .position(|anchor| anchor.transformer_variant.is_none())
        {
            let mut doctored: serde_json::Value =
                serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
            doctored["anchors"][axis_free]["transformerVariant"] = serde_json::json!("dev");
            let error = load_memory_anchors(&doctored.to_string())
                .expect_err("an invented pipeline variant must be rejected");
            assert!(error.contains("identity disagrees"), "{error}");
        }
    }

    #[test]
    fn a_duplicate_anchor_identity_is_rejected() {
        let ltx = ltx25_anchor_index();
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        let clone = doctored["anchors"][ltx].clone();
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
        let ltx = ltx25_anchor_index();
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        let peak = doctored["anchors"][ltx]["phaseActivePeakBytes"]["conditioning"]
            .as_u64()
            .expect("conditioning peak");
        doctored["anchors"][ltx]["phaseActivePeakBytes"]["conditioning"] =
            serde_json::json!(peak + 1);
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("a store whose bytes drift from the retained evidence must be rejected");
        assert!(error.contains("disagree"), "{error}");
    }

    #[test]
    fn a_stale_source_digest_or_foreign_source_path_is_rejected() {
        let ltx = ltx25_anchor_index();
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][ltx]["source"]["sha256"] = serde_json::json!("0".repeat(64));
        let error = load_memory_anchors(&doctored.to_string()).expect_err("digest must bind");
        assert!(error.contains("digest mismatch"), "{error}");

        let mut foreign: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        foreign["anchors"][ltx]["source"]["path"] = serde_json::json!("docs/nowhere.json");
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
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q8",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae,
            )
            .expect("q8 dev/diffvae anchor");
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
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q4",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae,
            )
            .expect("q4 dev/diffvae anchor");
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

    /// E3: coefficient uncertainty is priced INSIDE the coefficients, not inside the allocator
    /// margin. Every per-unit constant must be at or above the highest slope the retained corpus
    /// measures within one identity cell and regime — a coefficient sitting mid-spread would cross
    /// below the measured trend at large geometry, where nothing else would catch it.
    ///
    /// The corpus validation test cannot ask this question: its records all sit near the anchors,
    /// so a mid-spread coefficient still brackets them.
    #[test]
    fn every_per_unit_coefficient_bounds_the_highest_measured_within_cell_slope() {
        let corpus = retained_corpus();
        let mut cond_slopes: Vec<f64> = Vec::new();
        let mut denoise_slopes: Vec<f64> = Vec::new();
        let mut decode_slopes: Vec<f64> = Vec::new();
        for (index, left) in corpus.iter().enumerate() {
            for right in &corpus[index + 1..] {
                // Only a pair that differs in GEOMETRY ALONE measures a slope.
                if left.tier != right.tier
                    || left.transformer_variant != right.transformer_variant
                    || left.decoder != right.decoder
                    || left.decode_tiled != right.decode_tiled
                    || left.transformer_windowed != right.transformer_windowed
                    || left.deferred != right.deferred
                {
                    continue;
                }
                let left_tokens = latent_tokens(left.width, left.height, left.frames);
                let right_tokens = latent_tokens(right.width, right.height, right.frames);
                if left_tokens != right_tokens {
                    let delta = (right_tokens - left_tokens) as f64;
                    // Both intercept-priced phases; a bounded phase has no anchor slope to fit.
                    if !left.deferred {
                        cond_slopes.push(
                            (right.conditioning_active as f64 - left.conditioning_active as f64)
                                / delta,
                        );
                    }
                    denoise_slopes
                        .push((right.denoise_active as f64 - left.denoise_active as f64) / delta);
                }
                let left_voxels = voxels(left.width, left.height, left.frames);
                let right_voxels = voxels(right.width, right.height, right.frames);
                if left_voxels != right_voxels && !left.decode_tiled {
                    decode_slopes.push(
                        (right.decode_active as f64 - left.decode_active as f64)
                            / (right_voxels - left_voxels) as f64,
                    );
                }
            }
        }
        // Shape: the corpus must actually measure each slope, or this test asks nothing.
        assert!(!cond_slopes.is_empty(), "no measured conditioning slope");
        assert!(!denoise_slopes.is_empty(), "no measured denoise slope");
        assert!(!decode_slopes.is_empty(), "no measured decode slope");

        let highest = |slopes: &[f64]| slopes.iter().cloned().fold(f64::MIN, f64::max);
        for (name, coefficient, slopes) in [
            ("conditioning", COND_PER_TOKEN_BYTES as f64, &cond_slopes),
            ("denoise", DENOISE_PER_TOKEN_BYTES as f64, &denoise_slopes),
            ("decode", DECODE_PER_VOXEL_BYTES as f64, &decode_slopes),
        ] {
            let observed = highest(slopes);
            assert!(
                coefficient >= observed,
                "{name} coefficient {coefficient} sits below the highest measured within-cell \
                 slope {observed}: coefficient uncertainty must be priced in the coefficient, not \
                 in the allocator margin"
            );
        }
    }

    /// The frames term is affine in `frames`, and `frames == 0` must not underflow the latent
    /// frame count on its way there.
    #[test]
    fn latent_tokens_does_not_underflow_at_zero_frames() {
        assert_eq!(latent_tokens(640, 640, 0), latent_tokens(640, 640, 1));
    }

    /// An extrapolation driven non-positive has left the law's domain. Clamping it to zero would
    /// mint a 0-byte peak that admits ANYTHING, so widening refuses instead and the caller keeps
    /// its floor.
    ///
    /// No packaged anchor can currently be driven negative by geometry alone — the smallest
    /// derivable request still leaves every phase positive — so this asks the question of
    /// `widened` directly rather than pretending a reachable geometry exercises it. It is the
    /// guard that keeps a future anchor with a smaller intercept, or a larger coefficient, from
    /// silently minting a zero.
    #[test]
    fn a_non_positive_phase_estimate_is_refused_rather_than_clamped_to_zero() {
        assert_eq!(widened(-1), None);
        assert_eq!(widened(0), None);
        assert_eq!(widened(1_000_000), Some(1_170_000));
    }

    #[test]
    fn degenerate_geometry_is_not_derivable() {
        let anchor = store()
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q4",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae,
            )
            .expect("q4 dev/diffvae anchor");
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

        for record in &corpus {
            let anchor = store()
                .anchor_for(
                    "ltx_2_5",
                    AnchorBackend::Mlx,
                    &record.tier,
                    record.transformer_variant,
                    record.decoder,
                )
                .unwrap_or_else(|| {
                    panic!(
                        "cell ({}, {}, {}) carries an anchor",
                        record.tier,
                        transformer_variant_key(record.transformer_variant),
                        decoder_key(record.decoder)
                    )
                });
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

        // The tiled decode estimate must GROW with output voxels. The corpus has both of its
        // tiled captures at the same 130.7M output voxels, so this is the assertion that keeps a
        // geometry-blind flat constant from coming back: a flat bound would silently
        // under-admit a large tiled render whose output clip alone runs to gigabytes.
        let tiled_at = |voxels: i128| decode_tiled_bound_bytes(voxels);
        assert!(
            tiled_at(931_000_000) > tiled_at(130_662_400),
            "the tiled decode bound must grow with output voxels"
        );
        assert!(
            tiled_at(931_000_000) - tiled_at(130_662_400)
                >= DECODE_TILED_PER_VOXEL_BYTES * (931_000_000 - 130_662_400),
            "the tiled decode bound must grow at least at the output-clip rate"
        );

        // Corpus-support shape (informative, never a hard fail on the frozen corpus): the frames
        // and token terms are only validated where a cell carries a record that DIFFERS from its
        // anchor in latent token count. Where it does not — q4 today — the derivation's
        // extrapolation for that tier is unexercised, and this annotation is what makes the gap
        // visible instead of implied.
        for tier in ["q4", "q8", "bf16"] {
            let independent = corpus.iter().any(|record| {
                let Some(anchor) = store().anchor_for(
                    "ltx_2_5",
                    AnchorBackend::Mlx,
                    &record.tier,
                    record.transformer_variant,
                    record.decoder,
                ) else {
                    return false;
                };
                record.tier == tier
                    && record.id != anchor.source.record_id
                    && latent_tokens(record.width, record.height, record.frames)
                        != latent_tokens(
                            anchor.geometry.width,
                            anchor.geometry.height,
                            anchor.geometry.frames,
                        )
            });
            if !independent {
                eprintln!(
                    "corpus support gap: tier {tier} has no retained record that differs from its \
                     anchor in latent token count, so its token/frames extrapolation is validated \
                     by zero independent cells"
                );
            }
        }
    }
}
