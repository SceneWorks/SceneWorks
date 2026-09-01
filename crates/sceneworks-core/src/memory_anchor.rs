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
        "docs/generated/krea-candle-five-rung-sc-11045.json",
        include_str!("../../../docs/generated/krea-candle-five-rung-sc-11045.json"),
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
// Candle (discrete-VRAM image lane) derivation coefficients — sc-22509, epic 22505.
//
// The candle image lane is a DIFFERENT allocator and a different workload shape from the MLX video
// lane above, so it gets its own per-term coefficients rather than borrowing the LTX video ones:
//
// * There is no temporal axis. A still image has one latent frame, so every activation term scales
//   in OUTPUT PIXELS (`width x height`) rather than in latent tokens x latent frames. The law's
//   domain is bounded BELOW at the smallest retained measured geometry
//   ([`CANDLE_SMALLEST_RETAINED_PIXELS`]); a smaller request is priced at that geometry rather than
//   extrapolated into a region the corpus never touched.
// * There is no MLX-style reclaimable envelope. Every retained candle record measures
//   `cudaCachingAllocatorPresent = 0` and `reclaimableBytes = 0`, with `observedMemory.overall`
//   `allocatorBytes == activeBytes`: the CUDA lane hands pages back rather than retaining a cache
//   across phase transitions. [`CANDLE_ANCHOR_CAPTURE_SPREAD_MARGIN`] therefore covers a different
//   single term — capture-to-capture spread — and is named for it.
//
// PROVENANCE. The slopes below are the WITHIN-CELL measured deltas of the retained
// `krea_2_turbo` / candle / q4 / `threeStage` (`resident` + `staged_residency`) evidence pair at
// 768x768 and 1024x1024, published as `turboFit.evidenceRecords` in
// `config/manifests/builtin.models.jsonc` and re-measured per rung in the compiled-in
// `docs/generated/krea-candle-five-rung-sc-11045.json` capture the anchor itself is extracted from.
//
// The anchor sits at the TOP of the measured geometry range (1024x1024 is the largest measured
// still), so the corpus falsifies these coefficients by DOWNWARD extrapolation: a slope set too
// steep walks the estimate below the measured 768x768 peak. Each slope is therefore pinned below
// the ceiling that keeps the margin-widened 768x768 derivation at or above the measured 768x768
// peak, and at or above the physical growth term the architecture requires. `candle_anchor_
// derivation_brackets_every_retained_candle_measurement` is the falsifier for all of it.
// ---------------------------------------------------------------------------------------------

/// Conditioning-phase bytes per output pixel on the candle image lane. Conditioning under staged
/// residency holds the text encoder working set plus the initialized latent; only the latter sees
/// the image geometry. The Qwen-Image VAE is x8 spatial with 16 latent channels, so one fp32 latent
/// is `(w/8)(h/8) x 16 x 4 = 1` byte per output pixel; this is set at 4 B/px to cover up to four
/// concurrently live copies of it. The retained within-cell measured slope is NEGATIVE (3.565 GiB
/// at 768x768 against 3.44 GiB at 1024x1024 — the text working set is prompt-shaped, not
/// image-shaped), so any non-negative coefficient sits above the measured trend.
pub const CANDLE_COND_PER_PIXEL_BYTES: i128 = 4;

/// Denoise-phase bytes per output pixel on the candle image lane: the DiT forward's live activation
/// set over the packed image latent. Retained within-cell measured slope is 9,212 B/px
/// (10.597 -> 14.533 GiB across 589,824 -> 1,048,576 px on `q4` / `threeStage`). Set at 9 KiB/px,
/// which is above the measured slope for upward extrapolation and still below the 9,217 B/px
/// ceiling at which the margin-widened 768x768 derivation would fall under the measured peak. That
/// ceiling is the tightest of the three and is what pins
/// [`CANDLE_ANCHOR_CAPTURE_SPREAD_MARGIN`]'s value; both are checked by
/// `every_candle_coefficient_sits_inside_the_window_the_retained_pair_allows`.
pub const CANDLE_DENOISE_PER_PIXEL_BYTES: i128 = 9_216;

/// Decode-phase bytes per output pixel on the candle image lane: the VAE decoder's concurrently
/// live pixel-space working copies. Retained within-cell measured slope is 11,699 B/px
/// (16.286 -> 21.285 GiB over the same geometry pair). Set at 12 KiB/px — above the measured slope,
/// and below the 12,296 B/px ceiling that downward extrapolation to 768x768 imposes.
pub const CANDLE_DECODE_PER_PIXEL_BYTES: i128 = 12_288;

/// Smallest output area the retained candle corpus measures (768x768). Below it the derivation is
/// CLAMPED to this geometry rather than extrapolated further down: the corpus never touched that
/// region, and the linear slopes above — fitted across 768x768 -> 1024x1024 — walk the estimate below
/// the resident weight set the staged path still holds long before a 1x1 request, which errs toward
/// OOM. See [`MemoryAnchor::derive_image_phase_peaks`].
pub const CANDLE_SMALLEST_RETAINED_PIXELS: i128 = 768 * 768;

/// Multiplicative margin applied to every derived candle phase peak, at the widest point of the
/// derivation's domain. There is no MLX-style allocator-envelope term to cover here (the CUDA lane
/// reports `reclaimableBytes = 0` and no caching allocator in every retained record), and no blanket
/// safety factor: the value is the sum of two measured terms and nothing else.
///
/// TERM 1 — same-cell cross-capture spread, 3.3243%. The two retained candle captures of
/// `q4` / `staged_residency` / 1024x1024 disagree: denoise 15.1026 GB in the five-rung capture (the
/// anchor's own source record) against 15.6047 GB in the `turboFit` evidence record, i.e. +3.3243%;
/// decode 22.3525 against 22.8546 GB, i.e. +2.2463%. The anchor sits on the LOWER capture, so the
/// derivation must carry the wider of the two, 3.3243%.
///
/// TERM 2 — the downward-extrapolation lever, x1.3888. Term 1 is an ABSOLUTE discrepancy in the
/// anchor's INTERCEPT, but the margin multiplies the DERIVED value, which shrinks as the request
/// geometry falls below the anchor. At the clamp floor ([`CANDLE_SMALLEST_RETAINED_PIXELS`], the
/// widest point of the domain) the derived denoise base is
/// `15_102_640_128 - 9_216 x 458_752 = 10_874_781_696` bytes, or 0.72006 of the intercept, so the
/// same absolute discrepancy is `3.3243% / 0.72006 = 4.6168%` there. Decode's lever is weaker
/// (0.74781), so denoise sets the value.
///
/// TERM 3 — the deliberate slope surplus, +0.0169%. [`CANDLE_DENOISE_PER_PIXEL_BYTES`] is pinned
/// slightly ABOVE the measured 9,212 B/px slope so upward extrapolation stays conservative; that
/// same choice subtracts an extra `(9_216 - 9_212) x 458_752 = 1.835 MB` at the clamp floor, which
/// is 0.0169% of the base above.
///
/// `4.6168% + 0.0169% = 4.6337%`, rounded UP at the fourth decimal.
/// `candle_anchor_derivation_brackets_every_retained_candle_measurement` and
/// `candle_derivation_agrees_with_the_retained_measured_manifest_rows` are the falsifiers: the
/// binding requirement is the 768x768 denoise row, which needs 4.6315%.
pub const CANDLE_ANCHOR_CAPTURE_SPREAD_MARGIN: f64 = 0.0464;

/// Validation-only tightness budget for the candle lane, the sibling of
/// [`ANCHOR_VALIDATION_TIGHTNESS_BUDGET`]: the corpus validation test refuses a derived candle bound
/// more than this fraction above the measured peak of the anchor's OWN composition, so the
/// coefficients above cannot quietly widen into a vacuous always-passes bound. Deeper rung
/// compositions are only required to be BRACKETED, never to be tight: the shallow-anchor argument
/// (a deeper rung can only reduce a phase, never grow it) deliberately over-estimates them.
pub const CANDLE_ANCHOR_VALIDATION_TIGHTNESS_BUDGET: f64 = 0.15;

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
    /// Output rate of the anchor render, bound to the source record's `outputFps` measurement. The
    /// candle image lane emits a still and its records carry no such measurement, so this is
    /// `None` there and the validator requires the measurement to be absent in exactly that case.
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
    /// `staged_residency` was engaged for the anchor render. On the candle image lane this — not
    /// the record's `loadShape` — is the axis that decides whether the text encoder was still
    /// co-resident during conditioning, so it is load-bearing for
    /// [`MemoryAnchor::derive_image_phase_peaks`] rather than informational.
    pub staged: bool,
    /// `bounded_attention` was engaged for the anchor render.
    pub attention_chunked: bool,
}

/// One measured anchor: identity plus the peak decomposition of exactly one retained render.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryAnchor {
    pub id: String,
    pub model_id: String,
    /// Catalog family of `model_id`, carried for diagnostics and for the anchor-extraction receipt.
    ///
    /// NOT an identity conjunct and NOT authoritative: the source calibration record carries no
    /// family field, so this cannot be bound to the evidence the way every other identity axis is.
    /// The CATALOG (`models[].family` in the builtin manifest) is the authority, and the extractor
    /// populates this from it — an anchor lookup keys on `model_id` alone, which the record does
    /// bind. Never re-derive a guard from this field.
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
    /// derivable from this anchor. `None` on lanes whose pipeline has no such axis (the candle
    /// image lane): the validator requires the source record to agree, so an LTX anchor cannot
    /// silently drop them.
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

/// `None` is a real cell coordinate ("this lane has no such axis"), not a wildcard, so it gets its
/// own key spelling in the identity map and in the duplicate-anchor diagnostic.
fn optional_transformer_variant_key(variant: Option<Ltx25TransformerVariant>) -> &'static str {
    variant.map_or("-", transformer_variant_key)
}

fn optional_decoder_key(decoder: Option<Ltx25Decoder>) -> &'static str {
    decoder.map_or("-", decoder_key)
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
            optional_transformer_variant_key(anchor.transformer_variant),
            optional_decoder_key(anchor.decoder),
        );
        if let Some(previous) = identities.insert(identity, &anchor.id) {
            return Err(format!(
                "duplicate memory anchor for ({}, {}, {}, {}, {}): {} and {} — exactly one anchor \
                 per (model, tier, backend lane, transformer variant, decoder) is the schema \
                 invariant",
                anchor.model_id,
                anchor.tier,
                anchor.backend.as_key(),
                optional_transformer_variant_key(anchor.transformer_variant),
                optional_decoder_key(anchor.decoder),
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
    // `target.route` is absent from the LTX-2.5 seed; `video_memory_curves::source_route` treats
    // `target.provider` as the route spelling in exactly that case, and so does this.
    let record_route = str_at(target, "route").or_else(|| str_at(target, "provider"));
    if str_at(target, "modelId").as_deref() != Some(anchor.model_id.as_str())
        || str_at(target, "tier").as_deref() != Some(anchor.tier.as_str())
        || str_at(record, "backend").as_deref() != Some(anchor.backend.as_key())
        || str_at(target, "mode").as_deref() != Some(anchor.mode.as_str())
        || str_at(target, "provider").as_deref() != Some(anchor.provider.as_str())
        || record_route.as_deref() != Some(anchor.route.as_str())
        || str_at(target, "transformerVariant")
            != anchor
                .transformer_variant
                .map(|variant| transformer_variant_key(variant).to_owned())
        || str_at(target, "decoder")
            != anchor
                .decoder
                .map(|decoder| decoder_key(decoder).to_owned())
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
        || anchor.measured_regime.staged != engaged.contains(&"staged_residency")
        || anchor.measured_regime.attention_chunked != engaged.contains(&"bounded_attention")
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
    // Overlay and reference shape are identity conjuncts of both anchor lookups (`krea_store_anchor`
    // and `candle_image_anchor` refuse an overlaid or reference-bearing anchor), so they are bound
    // to the record rather than stored and trusted. The corpora spell "no overlay" as the string
    // `"none"`, and a record with no `referenceCount` measured a reference-free render.
    let record_overlay = str_at(target, "overlay")
        .filter(|overlay| overlay != "none")
        .filter(|overlay| !overlay.is_empty());
    let record_reference_count = geometry
        .get("referenceCount")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if record_overlay != anchor.overlay
        || record_reference_count != u64::from(anchor.reference_count)
    {
        return Err(format!(
            "memory anchor {} overlay/reference shape disagrees with its source record {}",
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
    // than leaving a serialized field nothing checks. A still-image lane record carries no
    // `outputFps`, and the anchor must then carry no fps — absence has to agree in both directions
    // or a video anchor could quietly drop the field.
    if measured.get("outputFps").copied() != anchor.geometry.fps.map(u64::from) {
        return Err(format!(
            "memory anchor {} fps {:?} disagrees with the outputFps measurement of its source \
             record {}",
            anchor.id, anchor.geometry.fps, anchor.source.record_id
        ));
    }
    // The two lanes name the same three phase peaks differently: the MLX adapter reports unified
    // ACTIVE peaks, the candle adapter reports discrete-device peak DELTAS. The names are chosen by
    // the anchor's backend rather than by probing for whichever is present, so a record captured on
    // one lane cannot satisfy an anchor declared on the other.
    let (conditioning_key, denoise_key, decode_key) = match anchor.backend {
        AnchorBackend::Mlx => (
            "conditioningActivePeak",
            "denoiseActivePeak",
            "decodeActivePeak",
        ),
        AnchorBackend::Candle => (
            "conditioningDevicePeakDelta",
            "denoiseDevicePeakDelta",
            "decodeDevicePeakDelta",
        ),
    };
    if measured.get(conditioning_key).copied() != Some(anchor.phase_active_peak_bytes.conditioning)
        || measured.get(denoise_key).copied() != Some(anchor.phase_active_peak_bytes.denoise)
        || measured.get(decode_key).copied() != Some(anchor.phase_active_peak_bytes.decode)
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
                && anchor.transformer_variant == Some(transformer_variant)
                && anchor.decoder == Some(decoder)
        })
    }

    /// The unique anchor for one `(model, backend lane, tier)` coordinate on a lane whose pipeline
    /// has no transformer-variant / decoder axis — the candle image lane (sc-22509).
    ///
    /// This is not a laxer spelling of [`Self::anchor_for`]: an anchor that DOES declare those axes
    /// is a different cell and is deliberately not returned here, so a video anchor can never be
    /// borrowed by an image request that simply omitted them.
    pub fn image_anchor_for(
        &self,
        model_id: &str,
        backend: AnchorBackend,
        tier: &str,
    ) -> Option<&MemoryAnchor> {
        self.anchors.iter().find(|anchor| {
            anchor.model_id == model_id
                && anchor.backend == backend
                && anchor.tier == tier
                && anchor.transformer_variant.is_none()
                && anchor.decoder.is_none()
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
    widened_by(bytes, ANCHOR_ALLOCATOR_ENVELOPE_MARGIN)
}

fn widened_by(bytes: i128, margin: f64) -> Option<u64> {
    if bytes <= 0 {
        return None;
    }
    let bytes = u64::try_from(bytes).ok()?;
    let widened = (bytes as f64 * (1.0 + margin)).ceil();
    (widened.is_finite() && widened < u64::MAX as f64).then_some(widened as u64)
}

/// The workload axes the candle image derivation prices. There is no temporal axis and no
/// per-phase rung flag: the anchor is the SHALLOWEST optimized composition the lane offers, and a
/// deeper rung can only reduce a phase below it (that is what the rung is for), so one law prices
/// every composition that contains the anchor's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorImageDeriveRequest {
    pub width: u32,
    pub height: u32,
    /// `staged_residency` engaged for the candidate being graded. A composition WITHOUT it runs the
    /// text encoder co-resident through denoise and decode, which the staged anchor does not price.
    pub staged_residency: bool,
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
        // Every coefficient in this law is an LTX-2.5 pipeline fact keyed on the transformer
        // variant and decoder. An anchor from a lane that has no such axes (the candle image
        // anchors, sc-22509) is refused here rather than silently priced by video coefficients.
        if self.transformer_variant.is_none() || self.decoder.is_none() {
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

    /// Derive the per-phase peak estimate for one requested still-image workload from this anchor
    /// (sc-22509, epic 22505 — the candle image-lane sibling of
    /// [`Self::derive_video_phase_peaks`]).
    ///
    /// Each phase is the anchor's measured intercept plus its own per-output-pixel term
    /// ([`CANDLE_COND_PER_PIXEL_BYTES`], [`CANDLE_DENOISE_PER_PIXEL_BYTES`],
    /// [`CANDLE_DECODE_PER_PIXEL_BYTES`]) applied to the pixel delta from the anchor geometry, then
    /// widened by [`CANDLE_ANCHOR_CAPTURE_SPREAD_MARGIN`].
    ///
    /// IDENTITY GUARD: this law is the CUDA image lane's. A non-candle anchor, an anchor carrying
    /// LTX pipeline axes, or an anchor whose measured geometry is not a still are all refused
    /// rather than coerced.
    ///
    /// REGIME GUARD: the anchor must be the SHALLOW optimized composition — `staged_residency`
    /// engaged and nothing deeper — and the graded candidate must also engage `staged_residency`.
    /// That asymmetry is the whole argument for pricing four rungs from one anchor: every deeper
    /// rung (`bounded_decode`, `bounded_attention`, `bounded_transformer_residency`) exists to make
    /// a phase SMALLER, so the shallow anchor upper-bounds them, while a resident composition holds
    /// the text encoder through denoise and decode and is strictly LARGER — the direction the
    /// anchor cannot cover. A resident request therefore keeps its own resident estimate.
    ///
    /// Returns `None` on degenerate geometry, on a regime or identity the anchor cannot price, and
    /// on any extrapolation that runs non-positive.
    pub fn derive_image_phase_peaks(
        &self,
        request: AnchorImageDeriveRequest,
    ) -> Option<AnchorDerivedPhases> {
        if request.width == 0 || request.height == 0 {
            return None;
        }
        if self.backend != AnchorBackend::Candle
            || self.transformer_variant.is_some()
            || self.decoder.is_some()
            || self.geometry.frames != 1
        {
            return None;
        }
        if !self.measured_regime.staged
            || self.measured_regime.decode_tiled
            || self.measured_regime.attention_chunked
            || self.measured_regime.transformer_windowed
            || !request.staged_residency
        {
            return None;
        }
        let anchor_pixels = i128::from(self.geometry.width) * i128::from(self.geometry.height);
        // LOWER CLAMP. The slopes are fitted across 768x768 -> 1024x1024 and the anchor sits at the
        // top of that range, so every smaller request is a DOWNWARD extrapolation. Continued below
        // the smallest retained geometry it leaves the corpus entirely and drops each phase far
        // under the resident weight set the staged path still holds (at 1x1 the derived decode is
        // ~5 GiB below the measured 768x768 decode) — an under-estimate, i.e. erring toward OOM. A
        // sub-768x768 request is therefore priced AT 768x768, which is the smallest bound the
        // evidence actually supports.
        let pixels = (i128::from(request.width) * i128::from(request.height))
            .max(CANDLE_SMALLEST_RETAINED_PIXELS);
        let delta = pixels - anchor_pixels;
        let conditioning = i128::from(self.phase_active_peak_bytes.conditioning)
            + CANDLE_COND_PER_PIXEL_BYTES * delta;
        let denoise = i128::from(self.phase_active_peak_bytes.denoise)
            + CANDLE_DENOISE_PER_PIXEL_BYTES * delta;
        let decode =
            i128::from(self.phase_active_peak_bytes.decode) + CANDLE_DECODE_PER_PIXEL_BYTES * delta;
        Some(AnchorDerivedPhases {
            conditioning: widened_by(conditioning, CANDLE_ANCHOR_CAPTURE_SPREAD_MARGIN)?,
            denoise: widened_by(denoise, CANDLE_ANCHOR_CAPTURE_SPREAD_MARGIN)?,
            decode: widened_by(decode, CANDLE_ANCHOR_CAPTURE_SPREAD_MARGIN)?,
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
                        && optional_transformer_variant_key(anchor.transformer_variant)
                            == *variant_key
                        && optional_decoder_key(anchor.decoder) == *decoder_key_str
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
        for anchor in &store().anchors {
            if anchor.model_id == "ltx_2_5" && anchor.backend == AnchorBackend::Mlx {
                assert_eq!(anchor.source.path, LTX25_CORPUS_PATH);
            }
        }
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
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][0]["loadShape"] = serde_json::json!("deferred_materialization");
        let error =
            load_memory_anchors(&doctored.to_string()).expect_err("load shape must bind to source");
        assert!(error.contains("load shape disagrees"), "{error}");

        // Every regime flag is bound, not just the two the video law reads: `staged` decides the
        // candle conditioning intercept and `attentionChunked` the denoise one.
        for field in [
            "decodeTiled",
            "transformerWindowed",
            "staged",
            "attentionChunked",
        ] {
            let mut doctored: serde_json::Value =
                serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
            let current = doctored["anchors"][0]["measuredRegime"][field]
                .as_bool()
                .unwrap_or_else(|| panic!("{field} is a declared regime flag"));
            doctored["anchors"][0]["measuredRegime"][field] = serde_json::json!(!current);
            let Err(error) = load_memory_anchors(&doctored.to_string()) else {
                panic!("{field} must bind to the engaged rungs");
            };
            assert!(
                error.contains("measured regime disagrees"),
                "{field}: {error}"
            );
        }
    }

    /// The candle anchor's own bound fields: the phase peaks are read under the CANDLE measurement
    /// names, and a still-image anchor may not invent an output rate.
    #[test]
    fn a_doctored_candle_anchor_measurement_binding_is_rejected() {
        let index = store()
            .anchors
            .iter()
            .position(|anchor| anchor.backend == AnchorBackend::Candle)
            .expect("the packaged store carries a candle anchor");
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][index]["geometry"]["fps"] = serde_json::json!(24);
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("a still-image anchor must not declare an fps its record never measured");
        assert!(error.contains("disagrees with the outputFps"), "{error}");

        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        let peak = doctored["anchors"][index]["phaseActivePeakBytes"]["denoise"]
            .as_u64()
            .expect("denoise peak");
        doctored["anchors"][index]["phaseActivePeakBytes"]["denoise"] = serde_json::json!(peak + 1);
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("the candle phase peaks must bind to the device-delta measurements");
        assert!(error.contains("peak bytes disagree"), "{error}");

        // The lane's measurement names are chosen by the anchor's backend, so relabelling the
        // anchor as MLX must fail rather than silently reading a different measurement set. The
        // record's own `backend` field is the FIRST conjunct that sees the relabel, so the exact
        // error is the identity one — asserting merely "disagree" would also have accepted the
        // peak-bytes failure and could not tell which guard fired.
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][index]["backend"] = serde_json::json!("mlx");
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("a relabelled lane must not resolve the other lane's measurements");
        assert!(error.contains("identity disagrees"), "{error}");

        // Overlay and reference shape are identity conjuncts of both anchor lookups, so they bind
        // to the record too rather than being stored and trusted.
        for (field, value) in [
            ("overlay", serde_json::json!("identity")),
            ("referenceCount", serde_json::json!(1)),
        ] {
            let mut doctored: serde_json::Value =
                serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
            doctored["anchors"][index][field] = value;
            let error = load_memory_anchors(&doctored.to_string())
                .err()
                .unwrap_or_else(|| panic!("{field} must bind to the source record"));
            assert!(
                error.contains("overlay/reference shape disagrees"),
                "{field}: {error}"
            );
        }
    }

    /// Identity fields that were previously stored-and-unchecked are bound to the source record.
    #[test]
    fn a_doctored_pipeline_identity_provider_route_or_fps_is_rejected() {
        for (field, value) in [
            ("transformerVariant", serde_json::json!("distilled")),
            ("decoder", serde_json::json!("conv")),
            ("provider", serde_json::json!("someone_else")),
            ("route", serde_json::json!("ltx_2_5_other")),
        ] {
            let mut doctored: serde_json::Value =
                serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
            doctored["anchors"][0][field] = value;
            let error = load_memory_anchors(&doctored.to_string())
                .err()
                .unwrap_or_else(|| panic!("{field} must bind to the source record"));
            assert!(error.contains("identity disagrees"), "{field}: {error}");
        }

        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        let fps = doctored["anchors"][0]["geometry"]["fps"]
            .as_u64()
            .expect("fps");
        doctored["anchors"][0]["geometry"]["fps"] = serde_json::json!(fps + 1);
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("fps must bind to the outputFps measurement");
        assert!(error.contains("disagrees with the outputFps"), "{error}");

        // Absence binds in the OTHER direction too: a video anchor may not drop the field its
        // record measured, which is the only way the still-image `None` spelling stays honest.
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][0]["geometry"]["fps"] = serde_json::Value::Null;
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("a measured outputFps may not be dropped from the anchor");
        assert!(error.contains("disagrees with the outputFps"), "{error}");
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

    // -------------------------------------------------------------------------------------
    // Candle image lane (sc-22509).
    // -------------------------------------------------------------------------------------

    const KREA_CANDLE_CORPUS_PATH: &str = "docs/generated/krea-candle-five-rung-sc-11045.json";

    /// One retained candle capture: the rung composition it executed and its three measured phase
    /// peaks.
    struct CandleCorpusRecord {
        id: String,
        engaged: Vec<String>,
        width: u32,
        height: u32,
        conditioning: u64,
        denoise: u64,
        decode: u64,
    }

    fn krea_candle_corpus() -> Vec<CandleCorpusRecord> {
        let raw = PACKAGED_MEMORY_ANCHOR_SOURCES
            .iter()
            .find(|(path, _)| *path == KREA_CANDLE_CORPUS_PATH)
            .map(|(_, raw)| *raw)
            .expect("the Krea candle retained corpus is compiled in");
        let source: serde_json::Value =
            serde_json::from_str(raw).expect("retained candle corpus parses");
        source["records"]
            .as_array()
            .expect("candle corpus records")
            .iter()
            .filter(|record| {
                record["target"]["modelId"].as_str() == Some("krea_2_turbo")
                    && record["backend"].as_str() == Some("candle")
            })
            .map(|record| {
                let geometry = &record["target"]["geometry"];
                let measured: BTreeMap<&str, u64> = record["diagnostics"]["measurements"]
                    .as_array()
                    .expect("measurements")
                    .iter()
                    .filter_map(|entry| Some((entry["name"].as_str()?, entry["value"].as_u64()?)))
                    .collect();
                CandleCorpusRecord {
                    id: record["id"].as_str().expect("record id").to_owned(),
                    engaged: record["strategy"]["engagedRungs"]
                        .as_array()
                        .expect("engaged rungs")
                        .iter()
                        .filter_map(|rung| rung.as_str().map(str::to_owned))
                        .collect(),
                    width: geometry["width"].as_u64().expect("width") as u32,
                    height: geometry["height"].as_u64().expect("height") as u32,
                    conditioning: measured["conditioningDevicePeakDelta"],
                    denoise: measured["denoiseDevicePeakDelta"],
                    decode: measured["decodeDevicePeakDelta"],
                }
            })
            .collect()
    }

    fn krea_candle_anchor() -> &'static MemoryAnchor {
        store()
            .image_anchor_for("krea_2_turbo", AnchorBackend::Candle, "q4")
            .expect("the Krea candle q4 anchor is packaged")
    }

    #[test]
    fn krea_candle_q4_carries_exactly_one_shallow_staged_anchor() {
        let anchor = krea_candle_anchor();
        // The candle image lane has no LTX pipeline axes, and its records carry no output rate.
        // Both are `None` as a positive statement about the cell, not an omission.
        assert_eq!(anchor.transformer_variant, None);
        assert_eq!(anchor.decoder, None);
        assert_eq!(anchor.geometry.fps, None);
        assert_eq!(anchor.geometry.frames, 1);
        assert_eq!(anchor.source.path, KREA_CANDLE_CORPUS_PATH);
        // The shallow staged composition is what makes one anchor price four rungs.
        assert_eq!(
            anchor.measured_regime,
            AnchorMeasuredRegime {
                decode_tiled: false,
                transformer_windowed: false,
                staged: true,
                attention_chunked: false,
            }
        );
        assert_eq!(
            store()
                .anchors
                .iter()
                .filter(|candidate| candidate.model_id == "krea_2_turbo"
                    && candidate.backend == AnchorBackend::Candle)
                .count(),
            1,
            "the retained candle corpus measures exactly one (model, tier, lane) cell"
        );
        // A video lookup must not reach an image anchor and vice versa.
        assert!(store()
            .anchor_for(
                "krea_2_turbo",
                AnchorBackend::Candle,
                "q4",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae,
            )
            .is_none());
        assert!(store()
            .image_anchor_for("ltx_2_5", AnchorBackend::Mlx, "bf16")
            .is_none());
    }

    /// AC 1 (corpus half): the candle derivation brackets every retained candle measurement whose
    /// composition contains the anchor's, and stays TIGHT on the anchor's own composition.
    #[test]
    fn candle_anchor_derivation_brackets_every_retained_candle_measurement() {
        let anchor = krea_candle_anchor();
        let corpus = krea_candle_corpus();
        assert!(
            corpus.len() >= 4,
            "the retained candle corpus must span several rung compositions, found {}",
            corpus.len()
        );
        let mut bracketed = 0usize;
        for record in &corpus {
            if !record.engaged.iter().any(|rung| rung == "staged_residency") {
                continue;
            }
            let derived = anchor
                .derive_image_phase_peaks(AnchorImageDeriveRequest {
                    width: record.width,
                    height: record.height,
                    staged_residency: true,
                })
                .unwrap_or_else(|| panic!("{} must be derivable", record.id));
            for (phase, derived_bytes, measured_bytes) in [
                ("conditioning", derived.conditioning, record.conditioning),
                ("denoise", derived.denoise, record.denoise),
                ("decode", derived.decode, record.decode),
            ] {
                assert!(
                    derived_bytes >= measured_bytes,
                    "{} {phase}: derived {derived_bytes} under-predicts the measured \
                     {measured_bytes}",
                    record.id
                );
                // NO tightness assert here. Every record in this corpus was captured at the
                // anchor's OWN geometry, so on the anchor's own composition the ratio is the
                // margin by construction (1.0 + CANDLE_ANCHOR_CAPTURE_SPREAD_MARGIN) and on a
                // deeper rung it is deliberately loose. Tightness is asserted where it can
                // actually bite — across BOTH measured geometries — by
                // `candle_derivation_agrees_with_the_retained_measured_manifest_rows`.
            }
            bracketed += 1;
        }
        assert!(
            bracketed >= 4,
            "at least four retained candle compositions must be bracketed, got {bracketed}"
        );
    }

    /// The regime guard is load-bearing, not decorative: the retained RESIDENT capture is the
    /// counterexample that proves reusing a staged anchor for a resident request would
    /// under-predict — and by how much.
    #[test]
    fn a_resident_candle_composition_is_refused_because_the_staged_anchor_underpredicts_it() {
        let anchor = krea_candle_anchor();
        assert!(
            anchor
                .derive_image_phase_peaks(AnchorImageDeriveRequest {
                    width: 1024,
                    height: 1024,
                    staged_residency: false,
                })
                .is_none(),
            "a resident composition must not be priced from the staged anchor"
        );
        let resident = krea_candle_corpus()
            .into_iter()
            .find(|record| record.engaged == ["resident"])
            .expect("the retained candle corpus carries a resident capture");
        let staged = anchor
            .derive_image_phase_peaks(AnchorImageDeriveRequest {
                width: resident.width,
                height: resident.height,
                staged_residency: true,
            })
            .expect("the staged derivation at the anchor geometry");
        assert!(
            staged.conditioning < resident.conditioning,
            "the resident conditioning peak {} must exceed the staged derivation {} — otherwise \
             the guard guards nothing",
            resident.conditioning,
            staged.conditioning
        );
        assert!(
            staged.decode < resident.decode,
            "the resident decode peak {} must exceed the staged derivation {}",
            resident.decode,
            staged.decode
        );
    }

    /// The other half of the regime guard: the ANCHOR must itself be the shallow staged capture.
    /// A capture measured in a deeper bounded regime carries a smaller working set than the rungs
    /// it would be asked to price, so it is refused rather than promoted into a law. The retained
    /// deeper captures are the counterexamples that make the refusal necessary.
    #[test]
    fn a_deeper_measured_candle_regime_cannot_be_promoted_into_the_shallow_law() {
        let request = AnchorImageDeriveRequest {
            width: 1024,
            height: 1024,
            staged_residency: true,
        };
        let shallow = krea_candle_anchor()
            .derive_image_phase_peaks(request)
            .expect("the shallow staged anchor prices its own geometry");
        for (label, regime, deeper_id) in [
            (
                "bounded_decode",
                AnchorMeasuredRegime {
                    decode_tiled: true,
                    transformer_windowed: false,
                    staged: true,
                    attention_chunked: false,
                },
                "imc-25e4d186c7016141e988",
            ),
            (
                "bounded_attention",
                AnchorMeasuredRegime {
                    decode_tiled: true,
                    transformer_windowed: false,
                    staged: true,
                    attention_chunked: true,
                },
                "imc-3fc9bf23d8b351edaa57",
            ),
            (
                "bounded_transformer_residency",
                AnchorMeasuredRegime {
                    decode_tiled: true,
                    transformer_windowed: true,
                    staged: true,
                    attention_chunked: true,
                },
                "imc-89d319fe2aed72ae01bb",
            ),
        ] {
            let mut mutated = krea_candle_anchor().clone();
            mutated.measured_regime = regime;
            assert!(
                mutated.derive_image_phase_peaks(request).is_none(),
                "an anchor measured under {label} must not price the shallow staged rung"
            );
            // …and the retained capture of that regime shows why: its denoise peak is genuinely
            // smaller than the shallow law's, so reusing it as an intercept would under-predict.
            let deeper = krea_candle_corpus()
                .into_iter()
                .find(|record| record.id == deeper_id)
                .unwrap_or_else(|| panic!("the retained {label} capture"));
            assert!(
                deeper.denoise < shallow.denoise,
                "{label}: the deeper capture's denoise {} must sit below the shallow derivation \
                 {} for the guard to be load-bearing",
                deeper.denoise,
                shallow.denoise
            );
        }
    }

    #[test]
    fn candle_derived_peaks_track_output_area_in_both_directions() {
        let anchor = krea_candle_anchor();
        let at = |width: u32, height: u32| {
            anchor
                .derive_image_phase_peaks(AnchorImageDeriveRequest {
                    width,
                    height,
                    staged_residency: true,
                })
                .unwrap_or_else(|| panic!("{width}x{height} must be derivable"))
        };
        let small = at(896, 896);
        let anchored = at(1024, 1024);
        let large = at(1536, 1536);
        assert!(small.peak_bytes() < anchored.peak_bytes());
        assert!(anchored.peak_bytes() < large.peak_bytes());
        // A never-measured non-square geometry is priced by area, not by aspect.
        assert_eq!(at(1344, 768).peak_bytes(), at(768, 1344).peak_bytes());
        // Below the smallest retained measured geometry the derivation CLAMPS instead of
        // extrapolating on: every request under 768x768 is priced at exactly 768x768, phase for
        // phase. Without the clamp the linear slopes walk the estimate below the working set the
        // staged path still holds, in a region the corpus never touched.
        let floor = at(768, 768);
        assert_eq!(
            CANDLE_SMALLEST_RETAINED_PIXELS,
            768 * 768,
            "the clamp floor is the smallest retained measured geometry"
        );
        for (label, clamped) in [("1x1", at(1, 1)), ("512x512", at(512, 512))] {
            assert_eq!(
                clamped, floor,
                "{label} must be priced at the 768x768 clamp floor, not extrapolated below it"
            );
        }
        // …and the floor is a real bound, not a degenerate one: it still exceeds the 768x768
        // measured decode peak of the retained manifest rows.
        let measured_768 = krea_retained_768_three_stage();
        assert!(floor.decode > measured_768.decode);
        assert!(floor.denoise > measured_768.denoise);
        assert!(anchor
            .derive_image_phase_peaks(AnchorImageDeriveRequest {
                width: 0,
                height: 512,
                staged_residency: true,
            })
            .is_none());
    }

    #[test]
    fn the_two_lane_laws_refuse_each_others_anchors() {
        assert!(
            krea_candle_anchor()
                .derive_video_phase_peaks(plain_request(1024, 1024, 1))
                .is_none(),
            "the LTX video law must refuse an anchor with no pipeline axes"
        );
        let ltx = store()
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "bf16",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae,
            )
            .expect("the LTX bf16 dev/diffvae anchor");
        assert!(
            ltx.derive_image_phase_peaks(AnchorImageDeriveRequest {
                width: 1024,
                height: 1024,
                staged_residency: true,
            })
            .is_none(),
            "the candle image law must refuse an MLX video anchor"
        );
        // The packaged LTX anchors are all attention-chunked, so the regime guard alone would
        // refuse them and the IDENTITY half of the guard would go unexercised. Each identity
        // condition is therefore mutated onto an otherwise-derivable candle anchor.
        let image_request = AnchorImageDeriveRequest {
            width: 1024,
            height: 1024,
            staged_residency: true,
        };
        assert!(
            krea_candle_anchor()
                .derive_image_phase_peaks(image_request)
                .is_some(),
            "the unmutated candle anchor must be derivable, or the mutations below prove nothing"
        );
        for (label, mutate) in [
            (
                "a video pipeline variant",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.transformer_variant = Some(Ltx25TransformerVariant::Dev);
                }) as Box<dyn Fn(&mut MemoryAnchor)>,
            ),
            (
                "a video decoder",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.decoder = Some(Ltx25Decoder::DiffVae);
                }),
            ),
            (
                "a multi-frame measured geometry",
                Box::new(|anchor: &mut MemoryAnchor| anchor.geometry.frames = 145),
            ),
            (
                "the other backend lane",
                Box::new(|anchor: &mut MemoryAnchor| anchor.backend = AnchorBackend::Mlx),
            ),
        ] {
            let mut mutated = krea_candle_anchor().clone();
            mutate(&mut mutated);
            assert!(
                mutated.derive_image_phase_peaks(image_request).is_none(),
                "the candle image law must refuse an anchor carrying {label}"
            );
        }
    }

    /// The `krea_2_turbo` catalog entry's `candle` block, from the embedded builtin manifest.
    fn krea_candle_manifest_block() -> serde_json::Value {
        let raw = crate::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, contents)| *contents)
            .expect("the builtin model manifest is embedded");
        let manifest: serde_json::Value =
            serde_json::from_str(&crate::jsonc::strip_jsonc_comments(raw))
                .expect("the builtin model manifest parses");
        manifest["models"]
            .as_array()
            .expect("models")
            .iter()
            .find(|model| model["id"].as_str() == Some("krea_2_turbo"))
            .expect("the krea_2_turbo catalog entry")["candle"]
            .clone()
    }

    /// The retained `krea_2_turbo` / candle / q4 / `threeStage` measured row at 768x768 — the
    /// SMALLEST retained geometry, and the one every downward-extrapolation bound is pinned
    /// against. Read from the manifest rather than transcribed, so a manifest edit cannot leave a
    /// stale literal validating a number the evidence no longer carries.
    fn krea_retained_768_three_stage() -> AnchorDerivedPhases {
        const GIB: f64 = 1_073_741_824.0;
        let candle = krea_candle_manifest_block();
        let row = candle["turboFit"]["evidenceRecords"]
            .as_array()
            .expect("turboFit evidence records")
            .iter()
            .find(|record| {
                record["tier"].as_str() == Some("q4")
                    && record["width"].as_u64() == Some(768)
                    && record["height"].as_u64() == Some(768)
            })
            .expect("the retained 768x768 q4 evidence record")["observedPhasesGb"]["threeStage"]
            .clone();
        let gib = |key: &str| (row[key].as_f64().unwrap_or_else(|| panic!("{key}")) * GIB) as u64;
        AnchorDerivedPhases {
            conditioning: gib("text"),
            denoise: gib("denoise"),
            decode: gib("decode"),
        }
    }

    /// AC 1 (manifest half): the derived-from-anchor estimates agree with the retained measured
    /// candle rows this story displaces as the selector's source — the `turboFit.evidenceRecords`
    /// per-phase captures at both measured geometries, and the `sequentialPeakGb` scalar the
    /// estimate floor used to read.
    #[test]
    fn candle_derivation_agrees_with_the_retained_measured_manifest_rows() {
        const GIB: f64 = 1_073_741_824.0;
        let anchor = krea_candle_anchor();
        let candle = krea_candle_manifest_block();
        let derive = |width: u32, height: u32| {
            anchor
                .derive_image_phase_peaks(AnchorImageDeriveRequest {
                    width,
                    height,
                    staged_residency: true,
                })
                .unwrap_or_else(|| panic!("{width}x{height} must be derivable"))
        };

        // Every retained q4 evidence record, at both measured geometries and all four measured
        // compositions.
        let records = candle["turboFit"]["evidenceRecords"]
            .as_array()
            .expect("turboFit evidence records");
        let mut checked = 0usize;
        for record in records {
            if record["tier"].as_str() != Some("q4") {
                continue;
            }
            let width = record["width"].as_u64().expect("width") as u32;
            let height = record["height"].as_u64().expect("height") as u32;
            let derived = derive(width, height);
            for (composition, phases) in record["observedPhasesGb"]
                .as_object()
                .expect("observed phases")
            {
                for (phase, derived_bytes) in [
                    ("text", derived.conditioning),
                    ("denoise", derived.denoise),
                    ("decode", derived.decode),
                ] {
                    let observed =
                        (phases[phase].as_f64().expect("observed phase") * GIB).ceil() as u64;
                    assert!(
                        derived_bytes >= observed,
                        "{width}x{height} {composition} {phase}: derived {derived_bytes} \
                         under-predicts the retained manifest row {observed}"
                    );
                    // Tight only against the anchor's own composition; deeper rungs are
                    // deliberately over-estimated.
                    if composition == "threeStage" {
                        let ratio = derived_bytes as f64 / observed as f64;
                        assert!(
                            ratio <= 1.0 + CANDLE_ANCHOR_VALIDATION_TIGHTNESS_BUDGET,
                            "{width}x{height} threeStage {phase}: derived {derived_bytes} is \
                             {ratio:.4}x the retained manifest row {observed}"
                        );
                    }
                }
                checked += 1;
            }
        }
        assert!(
            checked >= 8,
            "both retained q4 geometries and all four compositions must be checked, got {checked}"
        );

        // The scalar row the estimate floor used to read, at the geometry it was measured at.
        let measured_pixels = candle["vramMeasuredPixels"]
            .as_u64()
            .expect("vramMeasuredPixels");
        let edge = (measured_pixels as f64).sqrt() as u32;
        assert_eq!(u64::from(edge) * u64::from(edge), measured_pixels);
        assert_eq!(anchor.geometry.width, edge);
        let derived_peak = derive(edge, edge).peak_bytes();
        let sequential_row =
            (candle["sequentialPeakGb"]["q4"].as_f64().expect("row") * GIB).ceil() as u64;
        assert!(
            derived_peak >= sequential_row * 9 / 10 && derived_peak <= sequential_row * 11 / 10,
            "at the measured geometry the derived staged peak {derived_peak} must agree with the \
             retained sequentialPeakGb row {sequential_row} within 10%"
        );

        // Directional: the derivation prices the STAGED working set, so it must sit below the
        // resident row. If it ever crossed, the anchor would be pricing the wrong lane.
        let resident_row =
            (candle["vramGbByTier"]["q4"].as_f64().expect("row") * GIB).ceil() as u64;
        assert!(
            derived_peak < resident_row,
            "the derived staged peak {derived_peak} must stay below the retained resident row \
             {resident_row}"
        );
    }

    /// The candle slopes are pinned by DOWNWARD extrapolation: the anchor is the largest measured
    /// geometry, so a slope set too steep walks the estimate below the smaller measured peak. This
    /// pins each one against the ceiling that fact imposes, and against the physical growth term.
    #[test]
    fn every_candle_coefficient_sits_inside_the_window_the_retained_pair_allows() {
        // Retained `krea_2_turbo` / candle / q4 / `threeStage` pair, read from
        // `turboFit.evidenceRecords` — never transcribed as literals.
        let anchor = krea_candle_anchor();
        let measured_768 = krea_retained_768_three_stage();
        let anchor_pixels = i128::from(anchor.geometry.width) * i128::from(anchor.geometry.height);
        let span = anchor_pixels - CANDLE_SMALLEST_RETAINED_PIXELS;
        for (phase, intercept, small_measured, coefficient) in [
            (
                "conditioning",
                anchor.phase_active_peak_bytes.conditioning,
                measured_768.conditioning,
                CANDLE_COND_PER_PIXEL_BYTES,
            ),
            (
                "denoise",
                anchor.phase_active_peak_bytes.denoise,
                measured_768.denoise,
                CANDLE_DENOISE_PER_PIXEL_BYTES,
            ),
            (
                "decode",
                anchor.phase_active_peak_bytes.decode,
                measured_768.decode,
                CANDLE_DECODE_PER_PIXEL_BYTES,
            ),
        ] {
            assert!(
                coefficient >= 0,
                "{phase}: a negative per-pixel term would shrink the estimate as the image grows"
            );
            let small_measured = i128::from(small_measured);
            let ceiling = ((i128::from(intercept) as f64
                - small_measured as f64 / (1.0 + CANDLE_ANCHOR_CAPTURE_SPREAD_MARGIN))
                / span as f64) as i128;
            assert!(
                coefficient <= ceiling,
                "{phase}: coefficient {coefficient} B/px exceeds the {ceiling} B/px ceiling at \
                 which the margin-widened 768x768 derivation falls under the measured peak"
            );
        }
    }
}
