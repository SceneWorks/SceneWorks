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
        "docs/generated/krea-candle-five-rung-sc-11045.json",
        include_str!("../../../docs/generated/krea-candle-five-rung-sc-11045.json"),
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
// MLX (unified-memory) image lane derivation coefficients — epic 22505 feature-end fix round
// (E2/E7): the sibling of the candle image law above, for the six image-MLX anchors
// (flux2_dev q4/q8, qwen_image q4/q8/bf16, z_image_turbo q4).
//
// The MLX image lane differs from BOTH siblings in one measured way that shapes the whole law:
// the MLX allocator RETAINS CACHE ACROSS PHASE TRANSITIONS, and on an eager resident image render
// that retention is not a modest envelope above the binding phase — the retained flux2_dev
// captures show an overall allocator envelope up to 2.06x the largest phase ACTIVE peak (88.80 GB
// against a 43.18 GB decode active at q4/1024x1024). A per-phase law over ACTIVE peaks widened by
// a multiplicative margin therefore cannot honestly price admission here: the margin would have to
// be ~105% at one cell and ~43% at another, i.e. a number named after nothing. What IS linear in
// output pixels, per the retained within-cell pairs, is each phase's ALLOCATOR level — the active
// peak plus the cache retained from earlier phases, which is the quantity MLX admission must cover
// (the unified-memory budget is consumed by the allocator, not by the active set). So this law
// prices per-phase ALLOCATOR envelopes, read from the anchor's
// [`MemoryAnchor::phase_allocator_envelope_bytes`] (bound byte-exactly to the source record's
// `observedMemory.<phase>.allocatorBytes`), and its peak-over-phases IS the admission envelope —
// the decode-phase allocator level equals `observedMemory.overall.allocatorBytes` in every
// retained image-MLX record.
//
// PROVENANCE. The slopes are the within-cell measured allocator deltas of the retained
// `flux2_dev` MLX pairs at 768x768 -> 1024x1024 (eager, resident, no bounded rungs) in
// `docs/generated/memory-calibration-evidence.json`, at BOTH anchored tiers (q4 and q8) — the
// only image-MLX cells whose retained records vary geometry at all. Both anchors sit at the TOP
// of the measured range (1024x1024), so the corpus falsifies the coefficients by DOWNWARD
// extrapolation exactly as it does the candle law's: a slope set too steep walks the margin-
// widened 768x768 derivation below the measured 768x768 peak, and
// `every_mlx_image_coefficient_sits_inside_the_window_the_retained_pairs_allow` pins each one
// inside that window.
//
// PER-MODEL SCOPE. Coefficients fitted on flux2_dev's spread price flux2_dev's anchors and
// nothing else. qwen_image (all three tiers) and z_image_turbo retain records at a SINGLE
// geometry each (1024x1024 and 768x768 respectively), so no within-cell slope exists for them and
// none is borrowed: their anchors carry [`MemoryAnchor::underived_reason`] and this law refuses
// them — they validate their own measured point (the store handshake) and price nothing beyond
// it. That scoping is per-model by construction (the extractor computes it from each model's own
// retained geometry spread), never a blanket switch.
// ---------------------------------------------------------------------------------------------

/// Conditioning-phase allocator bytes per output pixel on the MLX image lane, expressed as ONE
/// BYTE PER THIS MANY PIXELS because the measured slope is far below a byte per pixel: the flux2
/// conditioning peak is the text-embedding working set (~1.2 MB), and its measured within-cell
/// allocator slope is 16,384 bytes over 458,752 px = 0.0357 B/px at both tiers. 1/16 B/px
/// (0.0625) sits above that slope for upward extrapolation and below the 0.0638 B/px ceiling at
/// which the margin-widened 768x768 derivation would fall under the measured 768x768 conditioning
/// allocator level.
pub const MLX_IMAGE_COND_ALLOC_PIXELS_PER_BYTE: i128 = 16;

/// Denoise-phase allocator bytes per output pixel on the MLX image lane: the DiT forward's live
/// activation growth (measured ACTIVE slope 2,259.9 B/px at both tiers) plus the allocator cache
/// retained across the phase. Measured within-cell allocator slopes: 7,514.4 and 7,738.7 B/px
/// (q8 pairs), 7,675.8 and 8,304.7 B/px (q4 pairs, the larger against the anchor's own capture).
/// Set at 8.25 KiB/px — above the highest measured slope, and below the 9,136 B/px ceiling the
/// q8 768x768 downward extrapolation imposes.
pub const MLX_IMAGE_DENOISE_ALLOC_PER_PIXEL_BYTES: i128 = 8_448;

/// Decode-phase allocator bytes per output pixel on the MLX image lane. The decode-phase
/// allocator level is the overall admission envelope (decode runs last and the pool still holds
/// the denoise-phase cache), so this is also the envelope's growth rate. Measured within-cell
/// slopes: 55,346.4 and 55,568.4 B/px (q8), 55,657.2 and 56,283.8 B/px (q4, the larger against
/// the anchor's own capture). Set at 55 KiB/px (56,320) — above the highest measured slope, and
/// below the 57,553 B/px ceiling the q8 768x768 downward extrapolation imposes.
pub const MLX_IMAGE_DECODE_ALLOC_PER_PIXEL_BYTES: i128 = 56_320;

/// Smallest output area the retained image-MLX corpus measures for the derivable models
/// (flux2_dev at 768x768). Below it the derivation CLAMPS to this geometry rather than
/// extrapolating on, for exactly the candle law's reason ([`CANDLE_SMALLEST_RETAINED_PIXELS`]):
/// the slopes are fitted across 768x768 -> 1024x1024 and walking further down leaves the corpus
/// entirely, in the under-estimate (OOM) direction.
pub const MLX_IMAGE_SMALLEST_RETAINED_PIXELS: i128 = 768 * 768;

/// Multiplicative margin applied to every derived MLX image phase allocator level. The allocator
/// envelope itself is NOT covered here — it is the derived quantity (see the section comment) —
/// so this margin covers exactly the RESIDUAL dispersion of that envelope that the upper-bounded
/// slopes cannot, i.e. the two measured terms left after each coefficient is pinned at or above
/// the highest within-cell slope:
///
/// * TERM 1 — cross-tier slope dispersion under downward extrapolation. The coefficients are
///   pinned above the q4 slopes (the steeper pair); at q8/768x768 the measured slopes are
///   shallower, so subtracting the pinned slope undershoots the measured level by 0.5301%
///   (denoise) and 0.3949% (decode).
/// * TERM 2 — the conditioning slope quantization. 1/16 B/px over the 458,752 px clamp span
///   subtracts 28,672 bytes where the corpus measured 16,384, leaving the 768x768 conditioning
///   derivation 1.0000% under its measured level — the binding requirement.
///
/// Set at 1.05%, above the 1.0000% binding term with rounding headroom.
/// `mlx_image_derivation_brackets_every_retained_flux2_measurement` is the falsifier.
pub const MLX_IMAGE_ALLOCATOR_RESIDUAL_MARGIN: f64 = 0.0105;

/// Validation-only tightness budget for the MLX image lane, the sibling of
/// [`ANCHOR_VALIDATION_TIGHTNESS_BUDGET`]: the corpus validation refuses a derived admission
/// bound more than this fraction above the record's measured overall allocator envelope, so the
/// coefficients and margin cannot quietly widen into a vacuous always-passes bound. 2% covers the
/// margin (1.05%) plus the observed same-cell capture spread of the 1024x1024 envelopes (0.33%).
pub const MLX_IMAGE_VALIDATION_TIGHTNESS_BUDGET: f64 = 0.02;

// ---------------------------------------------------------------------------------------------
// Variant/decoder component deltas — epic 22505 feature-end fix round (E2).
//
// An unmeasured `(transformer variant, decoder)` cell of an anchored `(model, tier, lane)` is
// priced from a SIBLING anchor of the same `(model, tier, lane)` plus deltas computed from the
// shipped weights file inventory: the variant's adapter/refiner file sizes and the decoder's
// weight file sizes, exactly as E2 states. The rule is deliberately one-directional and
// conservative:
//
//   * the delta ADDS the full file size of every component the TARGET cell materializes and the
//     sibling's cell does not. What each variant materializes is read off the ENGINE, not off the
//     variant's name: `mlx_ltx25.rs::configured_spec` pushes `DEV_ADAPTER` (the distillation LoRA)
//     only under `variant == Dev`, and `video_jobs/ltx.rs::resolve_ltx_distill_adapter` returns
//     `None` for LTX-2.5 Distilled because the distilled checkpoint already carries the
//     refinement. So the DEV row prices the LoRA, the DISTILLED row crosses nothing, and the stock
//     `enhancer/` directory — inserted into `spec.components` for BOTH variants — cancels out of
//     both. [`LTX25_VARIANT_DELTA_COMPONENTS`] mirrors that materialization and
//     `ltx25_variant_delta_matches_the_engine_materialization` cross-checks the store against it;
//     the decoder rows price the target decoder's weight files;
//   * a legitimately EMPTY crossing is carried as an explicit zero-byte row, not as a missing row.
//     The two say different things to [`MemoryAnchorStore::derive_video_phase_peaks_for_cell`]: a
//     missing row means "this axis is unpriced, refuse the sibling", a zero row means "crossing
//     this axis costs nothing", and collapsing them would make an honestly-free crossing
//     unreachable. A zero row still recomputes: it must name no files AND declare zero bytes, so a
//     nonzero row cannot be zeroed into one (its `files` would still sum nonzero);
//   * it subtracts NOTHING for components the sibling has and the target lacks. Subtracting
//     would assume the sibling's component was fully resident inside the phase peak being
//     re-priced — an assumption the retained evidence cannot certify (the MLX allocator retains
//     cache across phases, so a sibling-only component's bytes may or may not sit under the
//     measured level), and getting it wrong under-estimates, which is the OOM direction. Adding
//     only ever over-estimates.
//
// The byte values are BOUND, not trusted: every [`ComponentDelta`] row cites the shipped
// inventory file ([`PACKAGED_WEIGHTS_FILE_INVENTORIES`]) by digest and names the exact files it
// sums, and `validate_component_delta` recomputes the sum from the compiled-in inventory — a
// doctored or zeroed delta fails the load, and the zero-delta mutation reds
// `an_unmeasured_variant_cell_derives_from_the_sibling_anchor_plus_the_bound_delta`.
// ---------------------------------------------------------------------------------------------

/// Validation-only tightness budget for the sibling+delta fall-through, WIDER than
/// [`ANCHOR_VALIDATION_TIGHTNESS_BUDGET`] by design: the delta rule adds the FULL shipped file
/// size of every crossed component to every phase (the conservative direction — see the section
/// comment), so a derived off-anchor bound legitimately over-estimates by up to the component's
/// whole size where the true peak holds only part of it. The budget keeps the bound falsifiable
/// (the leave-one-out validation refuses a runaway estimate) without demanding a tightness the
/// conservative rule cannot deliver.
///
/// RE-DERIVED at the feature-end fix round, after the variant deltas were re-keyed onto what each
/// variant actually materializes (the earlier 0.35 was sized by a binding case that inherited a
/// misattributed 8.9 GB distillation-LoRA add on the distilled row — a component the distilled
/// target never loads). With the corrected table the binding retained case is the bf16
/// distilled/conv 1280x704 capture `imc-4e1b3a02a6ced3434824`, derived from its dev/diffvae
/// sibling across a zero variant crossing plus the 0.81 GB conv-decoder crossing, landing 21.83%
/// over its measured envelope. The budget is that figure rounded up to the next whole point.
/// `the_delta_tightness_budget_is_the_binding_leave_one_out_overshoot_rounded_up` recomputes it
/// and refuses more than a point of slack, so this cannot drift back into a number nothing binds.
pub const ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET: f64 = 0.22;

/// Shipped weights file inventories the component deltas cite, compiled in for the same reason
/// the retained corpora are: a delta's byte value must be recomputable from bytes this build
/// carries, not from whatever the file on disk currently says. The inventory records per-file
/// sizes of a pinned artifact revision (immutable content, so the sizes cannot drift).
pub const PACKAGED_WEIGHTS_FILE_INVENTORIES: &[(&str, &str)] = &[(
    "config/ltx25-weights-file-inventory.json",
    include_str!("../../../config/ltx25-weights-file-inventory.json"),
)];

/// What each LTX-2.5 transformer variant materializes beyond the components its sibling ALSO
/// loads, keyed by the component's top-level directory in the shipped bundle.
///
/// This is a MIRROR of the engine, held here because `sceneworks-core` cannot depend on the worker
/// or the capture arm, and because a variant delta keyed on anything other than the engine's real
/// materialization is silently wrong in the OOM direction. The two authorities:
///
///   * `crates/sceneworks-memory-adapter/src/bin/mlx_ltx25.rs::configured_spec` pushes
///     `DEV_ADAPTER` — the `distilled_lora/` file — onto `spec.adapters` only under
///     `variant == TransformerVariant::Dev`, and inserts the `enhancer` component for BOTH
///     variants (so the enhancer cancels and is not listed here for either);
///   * `crates/sceneworks-worker/src/video_jobs/ltx.rs::resolve_ltx_distill_adapter` returns
///     `None` for `Ltx25TransformerVariant::Distilled`, whose checkpoint already carries the
///     refinement, and resolves the manifest's `distilledLora` co-requisite for dev.
///
/// `ltx25_variant_delta_components_mirror_the_engine_source` cross-checks this table against those
/// two source files, and `ltx25_variant_delta_matches_the_engine_materialization` checks the
/// shipped delta rows against this table — so re-inverting the mapping reds, in either place.
pub const LTX25_VARIANT_DELTA_COMPONENTS: &[(&str, &[&str])] =
    &[("dev", &["distilled_lora"]), ("distilled", &[])];

/// The pipeline axis a component delta re-prices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentDeltaAxis {
    TransformerVariant,
    Decoder,
}

/// Where a delta's byte value comes from: a shipped inventory file, cited by digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ComponentDeltaSource {
    /// Repo-relative path of the inventory; must be a compiled-in inventory.
    pub path: String,
    /// SHA-256 of that file's bytes at extraction time.
    pub sha256: String,
}

/// One `(model, lane, tier, axis, target value)` component delta: the bytes added when deriving a
/// cell whose `axis` equals `to` from a sibling anchor whose `axis` differs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ComponentDelta {
    pub id: String,
    pub model_id: String,
    pub backend: AnchorBackend,
    pub tier: String,
    pub axis: ComponentDeltaAxis,
    /// The target cell's value on `axis` (store-key spelling, e.g. `distilled` / `conv`).
    pub to: String,
    /// The summed size of `files`, recomputed from the cited inventory at load.
    pub bytes: u64,
    /// The inventory paths summed into `bytes` — the component's shipped weight files.
    pub files: Vec<String>,
    pub source: ComponentDeltaSource,
}

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
    /// Variant/decoder component deltas (epic 22505 E2, feature-end fix round): the bound byte
    /// costs the sibling-anchor fall-through of
    /// [`MemoryAnchorStore::derive_video_phase_peaks_for_cell`] adds for an unmeasured pipeline
    /// cell. `default` so a store written before the migration still parses.
    #[serde(default)]
    pub component_deltas: Vec<ComponentDelta>,
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
    /// The calibration campaign that produced the record. PROVENANCE, not currency: since sc-22511
    /// an anchor's currency is [`AnchorSource::loader_closure_digest`] and nothing else, so a new
    /// campaign fingerprint no longer demotes evidence whose loader never moved. It stays bound to
    /// the source record by [`validate_anchor`] so the anchor cannot misattribute its own origin.
    pub calibration_fingerprint: String,
    /// THE CURRENCY KEY (sc-22511, epic 22505 E9): the digest of the model's OWN loader closure —
    /// the source files that load and execute this model on this backend — at the revision the
    /// anchor was measured at. Derived by `scripts/anchor-loader-closure.mjs`, whose module comment
    /// owns the definition of the unit and what it deliberately does not see.
    ///
    /// The anchor is CURRENT while this still equals the digest declared for its
    /// `(model, backend lane)` in [`PACKAGED_ANCHOR_LOADER_CLOSURES`]. Because the unit contains no
    /// revision, no lock and no workspace input, a pin bump that leaves the loader's source
    /// untouched leaves this equal — which is exactly the claim E9 makes: an anchor predating an
    /// unrelated change stays authoritative.
    pub loader_closure_digest: String,
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
    /// The measured per-phase ALLOCATOR levels of the anchor render (active + retained cache at
    /// each phase's peak), bound byte-exactly to the source record's
    /// `observedMemory.<phase>.allocatorBytes` in BOTH directions: a record that reports all
    /// three phase allocator levels must have them here, and an anchor may not invent them.
    /// This is the quantity the MLX image law derives ([`MemoryAnchor::derive_mlx_image_phase_peaks`])
    /// — see that section's comment for why actives cannot price that lane. `None` where the
    /// source record reports no complete per-phase allocator decomposition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_allocator_envelope_bytes: Option<AnchorPhaseBytes>,
    /// The measured overall allocator envelope of the anchor render (active + reclaimable).
    pub overall_allocator_envelope_bytes: u64,
    /// `Some` when this anchor VALIDATES its measured point but no lane law may derive from it,
    /// with the stated reason (epic 22505 feature-end fix round, per-model scoping). Every
    /// derivation law refuses an anchor carrying this field; the memory matrix publishes the cell
    /// as `Anchored/underived`. Written by the extractor from the model's own retained evidence
    /// (e.g. a single measured geometry supports no per-pixel coefficient), never a blanket
    /// switch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub underived_reason: Option<String>,
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
    validate_component_deltas(&store)?;
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
    // Shape only. A MISMATCHED loader digest is a staleness verdict at admission, never a load
    // failure: the store keeps carrying the evidence it measured, and the derivation declines to
    // price with it until the digest agrees again.
    if !is_sha256(&anchor.source.loader_closure_digest) {
        return Err(format!(
            "memory anchor {} loader closure digest {} is not a sha256",
            anchor.id, anchor.source.loader_closure_digest
        ));
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
    // Per-phase ALLOCATOR levels (epic 22505 feature-end fix round): bound in BOTH directions to
    // `observedMemory.<phase>.allocatorBytes`, so a record that reports the decomposition must
    // carry it and an anchor cannot invent or doctor one. The MLX image law derives from these.
    let phase_allocator = |phase: &str| {
        record
            .get("observedMemory")
            .and_then(|memory| memory.get(phase))
            .and_then(|entry| entry.get("allocatorBytes"))
            .and_then(serde_json::Value::as_u64)
            .filter(|&bytes| bytes > 0)
    };
    let record_allocators = match (
        phase_allocator("conditioning"),
        phase_allocator("denoise"),
        phase_allocator("decode"),
    ) {
        (Some(conditioning), Some(denoise), Some(decode)) => Some(AnchorPhaseBytes {
            conditioning,
            denoise,
            decode,
        }),
        _ => None,
    };
    if record_allocators != anchor.phase_allocator_envelope_bytes {
        return Err(format!(
            "memory anchor {} phase allocator envelopes disagree with the observedMemory of its \
             source record {} — the field is bound in both directions",
            anchor.id, anchor.source.record_id
        ));
    }
    if anchor
        .underived_reason
        .as_ref()
        .is_some_and(|reason| reason.trim().is_empty())
    {
        return Err(format!(
            "memory anchor {} declares itself underived without a reason — an unexplained \
             refusal is a gap wearing a field's clothes",
            anchor.id
        ));
    }
    Ok(())
}

/// The component-delta half of the store's invariants (epic 22505 E2, feature-end fix round).
///
/// Every delta's byte value is RECOMPUTED from the compiled-in weights file inventory it cites, so
/// the store cannot carry a delta the shipped inventory does not entail — a zeroed, inflated or
/// re-pointed row fails the load rather than silently re-pricing a cell.
fn validate_component_deltas(store: &MemoryAnchorStore) -> Result<(), String> {
    let mut seen = BTreeMap::new();
    for delta in &store.component_deltas {
        let key = (
            delta.model_id.clone(),
            delta.backend,
            delta.tier.clone(),
            delta.axis,
            delta.to.clone(),
        );
        if let Some(previous) = seen.insert(key, &delta.id) {
            return Err(format!(
                "duplicate component delta for ({}, {}, {}, {:?}, {}): {} and {}",
                delta.model_id,
                delta.backend.as_key(),
                delta.tier,
                delta.axis,
                delta.to,
                previous,
                delta.id
            ));
        }
        // A row that names no files is the ZERO delta: the target crosses no component the sibling
        // lacks. It is a real claim, not a hole, so it is representable — but only as a genuine
        // zero. Declaring bytes without files is the doctoring case and still fails.
        if delta.files.is_empty() && delta.bytes != 0 {
            return Err(format!(
                "component delta {} declares {} bytes but names no files — only a zero-byte \
                 crossing may name none",
                delta.id, delta.bytes
            ));
        }
        let Some((_, raw)) = PACKAGED_WEIGHTS_FILE_INVENTORIES
            .iter()
            .find(|(path, _)| *path == delta.source.path)
        else {
            return Err(format!(
                "component delta {} cites {} which is not a compiled weights file inventory",
                delta.id, delta.source.path
            ));
        };
        let digest = format!("{:x}", Sha256::digest(raw.as_bytes()));
        if digest != delta.source.sha256 {
            return Err(format!(
                "component delta {} inventory digest mismatch for {}: recorded {} actual {digest}",
                delta.id, delta.source.path, delta.source.sha256
            ));
        }
        let inventory: serde_json::Value = serde_json::from_str(raw).map_err(|error| {
            format!(
                "weights inventory {} does not parse: {error}",
                delta.source.path
            )
        })?;
        let files = inventory
            .get("files")
            .and_then(|files| files.as_object())
            .ok_or_else(|| format!("weights inventory {} has no files map", delta.source.path))?;
        let mut total: u64 = 0;
        for file in &delta.files {
            let size = files
                .get(file)
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| {
                    format!(
                        "component delta {} sums {} which the inventory {} does not size",
                        delta.id, file, delta.source.path
                    )
                })?;
            // Keeps the zero-row rule above airtight: a row that names files must sum positive,
            // so "no files" stays the ONLY way a delta reaches zero.
            if size == 0 {
                return Err(format!(
                    "component delta {} sums {} which the inventory {} sizes at zero",
                    delta.id, file, delta.source.path
                ));
            }
            total = total.saturating_add(size);
        }
        // `total == 0` is NOT rejected on its own: with no files it is the honest zero crossing
        // (guarded above), and a row that names files cannot reach zero (guarded just above).
        // What is rejected is a declared byte count the cited files do not entail — which is what
        // a zeroed, inflated or re-pointed row looks like.
        if total != delta.bytes {
            return Err(format!(
                "component delta {} declares {} bytes but its cited files sum to {total} — the \
                 delta must equal the shipped file sizes it names",
                delta.id, delta.bytes
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Anchor currency: the model's own loader closure (sc-22511).
// ---------------------------------------------------------------------------------------------

/// The checked-in loader-closure declarations — the CURRENT value of every declared model's
/// currency key. Derived at the pinned inference revision by `scripts/anchor-loader-closure.mjs`.
pub const PACKAGED_ANCHOR_LOADER_CLOSURES: &str =
    include_str!("../../../config/anchor-loader-closures.json");

/// Must equal the `digestVersion` of the checked-in file. Two versions answer different questions,
/// so a version bump reads as "no declaration" (fail closed to the floor) rather than silently
/// comparing digests derived under different rules.
pub const ANCHOR_LOADER_CLOSURE_VERSION: &str = "anchor-loader-closure v2";

/// One `(model, backend lane)`'s declared loader closure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnchorLoaderClosure {
    /// The loader entry points the closure is rooted at, repo-relative in the inference tree.
    pub entry_points: Vec<String>,
    /// The digest of the closure's source content. This is what an anchor is compared against.
    pub digest: String,
    pub closure_file_count: usize,
    /// The resolved file list, checked in so a digest change is answerable — "what moved?" has to
    /// be readable from the diff without re-deriving anything.
    pub closure_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnchorLoaderClosures {
    #[serde(rename = "_comment", default)]
    pub comment: String,
    pub digest_version: String,
    /// The revision the digests were DERIVED at. Provenance only — deliberately never compared,
    /// for the same reason the digest does not hash it.
    pub inference_revision: String,
    /// Keyed `"<model id>:<backend lane>"`.
    pub models: BTreeMap<String, AnchorLoaderClosure>,
}

/// The declaration key for one anchor coordinate.
pub fn anchor_loader_closure_key(model_id: &str, backend: AnchorBackend) -> String {
    format!("{model_id}:{}", backend.as_key())
}

impl AnchorLoaderClosures {
    /// The current loader-closure digest for one `(model, backend lane)`, or `None` when the
    /// coordinate is undeclared — which is fail-closed: an anchor whose loader nothing tracks
    /// cannot be shown to be current, so it is not.
    pub fn digest_for(&self, model_id: &str, backend: AnchorBackend) -> Option<&str> {
        self.models
            .get(&anchor_loader_closure_key(model_id, backend))
            .map(|closure| closure.digest.as_str())
    }
}

/// Strict parse plus the invariants serde cannot express.
pub fn load_anchor_loader_closures(raw: &str) -> Result<AnchorLoaderClosures, String> {
    let closures: AnchorLoaderClosures = serde_json::from_str(raw)
        .map_err(|error| format!("anchor loader closures do not parse: {error}"))?;
    if closures.digest_version != ANCHOR_LOADER_CLOSURE_VERSION {
        return Err(format!(
            "anchor loader closure digest version {} is not the supported \
             {ANCHOR_LOADER_CLOSURE_VERSION}",
            closures.digest_version
        ));
    }
    for (key, closure) in &closures.models {
        if closure.entry_points.is_empty() {
            return Err(format!(
                "anchor loader closure {key} declares no entry points"
            ));
        }
        if !is_sha256(&closure.digest) {
            return Err(format!(
                "anchor loader closure {key} digest {} is not a sha256",
                closure.digest
            ));
        }
        if closure.closure_files.len() != closure.closure_file_count {
            return Err(format!(
                "anchor loader closure {key} lists {} files but declares {}",
                closure.closure_files.len(),
                closure.closure_file_count
            ));
        }
        // An entry point outside the walked closure would mean the digest was derived from a
        // different root than the one declared here.
        for entry in &closure.entry_points {
            if !closure.closure_files.contains(entry) {
                return Err(format!(
                    "anchor loader closure {key} roots at {entry}, which is not in its own closure"
                ));
            }
        }
    }
    Ok(closures)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// The packaged declarations, parsed once. `None` demotes every anchor to the caller's floor.
pub fn packaged_anchor_loader_closures() -> Option<&'static AnchorLoaderClosures> {
    static PACKAGED: OnceLock<Option<AnchorLoaderClosures>> = OnceLock::new();
    PACKAGED
        .get_or_init(|| load_anchor_loader_closures(PACKAGED_ANCHOR_LOADER_CLOSURES).ok())
        .as_ref()
}

impl MemoryAnchor {
    /// Whether this anchor's evidence is CURRENT (sc-22511, E9).
    ///
    /// The one and only currency question: does the code that loads THIS model on THIS backend
    /// still hash to what it hashed when the anchor was measured? Not the pin, not a sibling model,
    /// not a shared crate the loader never reaches, not the calibration campaign that produced the
    /// record — none of those can move this answer, by construction of the key.
    pub fn is_current(&self, closures: &AnchorLoaderClosures) -> bool {
        closures.digest_for(&self.model_id, self.backend)
            == Some(self.source.loader_closure_digest.as_str())
    }
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

/// One phase's raw (pre-margin) video estimate: the bytes, and whether they reuse the anchor's
/// OWN measured intercept (`anchored`) rather than a cell-blind architecture bound — the flag the
/// sibling+delta fall-through keys its adds on (see `derive_video_phase_estimates_raw`).
#[derive(Debug, Clone, Copy)]
struct RawPhaseEstimate {
    bytes: i128,
    anchored: bool,
}

/// The three raw phase estimates of one video derivation.
#[derive(Debug, Clone, Copy)]
struct RawVideoPhaseEstimates {
    conditioning: RawPhaseEstimate,
    denoise: RawPhaseEstimate,
    decode: RawPhaseEstimate,
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
        let raw = self.derive_video_phase_estimates_raw(request)?;
        Some(AnchorDerivedPhases {
            conditioning: widened(raw.conditioning.bytes)?,
            denoise: widened(raw.denoise.bytes)?,
            decode: widened(raw.decode.bytes)?,
        })
    }

    /// The video law's raw (pre-margin) per-phase estimates, shared by the exact-cell path above
    /// and the sibling+delta fall-through ([`MemoryAnchorStore::derive_video_phase_peaks_for_cell`],
    /// which adds its component delta BEFORE the shared margin is applied). Each phase carries an
    /// `anchored` flag: `true` when the estimate reuses the anchor's OWN measured intercept (and
    /// therefore embeds the anchor cell's component set — the part a crossed-axis delta must
    /// re-price), `false` when it is an architecture bound
    /// ([`COND_DEFERRED_BOUND_BYTES`] / [`DENOISE_WINDOWED_BASE_BYTES`] /
    /// [`decode_tiled_bound_bytes`]), which is cell-blind by construction and documented to
    /// upper-bound every variant/decoder already.
    fn derive_video_phase_estimates_raw(
        &self,
        request: AnchorDeriveRequest,
    ) -> Option<RawVideoPhaseEstimates> {
        if request.width == 0 || request.height == 0 || request.frames == 0 {
            return None;
        }
        // An anchor marked underived validates its measured point and prices nothing (epic 22505
        // feature-end fix round): every lane law carries this refusal so the store's statement and
        // the runtime behaviour cannot disagree.
        if self.underived_reason.is_some() {
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

        let phase = |bytes: i128, anchored: bool| RawPhaseEstimate { bytes, anchored };
        let conditioning = if request.deferred_materialization {
            phase(i128::from(COND_DEFERRED_BOUND_BYTES), false)
        } else {
            phase(
                i128::from(self.phase_active_peak_bytes.conditioning)
                    + COND_PER_TOKEN_BYTES * (tokens - anchor_tokens),
                true,
            )
        };
        let denoise = if request.transformer_windowed {
            phase(
                i128::from(DENOISE_WINDOWED_BASE_BYTES) + DENOISE_PER_TOKEN_BYTES * tokens,
                false,
            )
        } else {
            phase(
                i128::from(self.phase_active_peak_bytes.denoise)
                    + DENOISE_PER_TOKEN_BYTES * (tokens - anchor_tokens),
                true,
            )
        };
        let decode = if request.decode_tiled {
            phase(decode_tiled_bound_bytes(voxels), false)
        } else {
            phase(
                i128::from(self.phase_active_peak_bytes.decode)
                    + DECODE_PER_VOXEL_BYTES * (voxels - anchor_voxels),
                true,
            )
        };
        Some(RawVideoPhaseEstimates {
            conditioning,
            denoise,
            decode,
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
        // See `derive_video_phase_estimates_raw`: an underived anchor prices nothing on any lane.
        if self.underived_reason.is_some() {
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

    /// Derive the per-phase ALLOCATOR levels for one requested still-image workload on the MLX
    /// unified-memory lane (epic 22505 feature-end fix round, E2/E7 — the MLX sibling of
    /// [`Self::derive_image_phase_peaks`]). See the coefficient section's comment for why this
    /// lane prices phase ALLOCATOR envelopes rather than actives; the returned
    /// [`AnchorDerivedPhases::peak_bytes`] is therefore directly the admission envelope, and the
    /// selector adds NOTHING on top (the worker's ladder margin policy treats
    /// `EstimateAnchorDerived` as fully priced).
    ///
    /// IDENTITY GUARD: this law is the MLX image lane's. A non-MLX anchor, an anchor carrying LTX
    /// pipeline axes, a multi-frame anchor, or an anchor whose record reports no per-phase
    /// allocator decomposition are refused rather than coerced. An anchor carrying
    /// [`MemoryAnchor::underived_reason`] is refused for the reason it states.
    ///
    /// REGIME GUARD: the anchor must be the fully UNBOUNDED eager resident composition — no rung
    /// engaged, eager materialization. That is the WIDEST composition the lane executes, so it
    /// upper-bounds every optimized composition of the same cell (each rung exists to make a
    /// phase smaller) and one law prices the whole ladder. This is the mirror image of the candle
    /// law's staged-anchor asymmetry: there the shallow OPTIMIZED anchor could not price the
    /// larger resident composition; here the resident anchor prices everything because nothing is
    /// larger than it.
    ///
    /// Returns `None` on degenerate geometry and on any identity/regime the anchor cannot price.
    pub fn derive_mlx_image_phase_peaks(
        &self,
        request: AnchorMlxImageDeriveRequest,
    ) -> Option<AnchorDerivedPhases> {
        if request.width == 0 || request.height == 0 {
            return None;
        }
        if self.underived_reason.is_some() {
            return None;
        }
        if self.backend != AnchorBackend::Mlx
            || self.transformer_variant.is_some()
            || self.decoder.is_some()
            || self.geometry.frames != 1
        {
            return None;
        }
        if self.load_shape != AnchorLoadShape::EagerMaterialization
            || self.measured_regime.staged
            || self.measured_regime.decode_tiled
            || self.measured_regime.attention_chunked
            || self.measured_regime.transformer_windowed
        {
            return None;
        }
        let allocators = self.phase_allocator_envelope_bytes?;
        let anchor_pixels = i128::from(self.geometry.width) * i128::from(self.geometry.height);
        // LOWER CLAMP, exactly as the candle law's: the slopes are fitted across
        // 768x768 -> 1024x1024 with the anchor at the top, so a sub-768x768 request is priced AT
        // 768x768 rather than extrapolated below the corpus.
        let pixels = (i128::from(request.width) * i128::from(request.height))
            .max(MLX_IMAGE_SMALLEST_RETAINED_PIXELS);
        let delta = pixels - anchor_pixels;
        // The conditioning slope is sub-byte-per-pixel, so it is applied as `delta / 16` with the
        // rounding chosen conservative in BOTH directions: an upward delta rounds UP (charge at
        // least the slope), a downward delta rounds toward zero (subtract at most the slope).
        let conditioning_growth = if delta >= 0 {
            (delta + MLX_IMAGE_COND_ALLOC_PIXELS_PER_BYTE - 1)
                .div_euclid(MLX_IMAGE_COND_ALLOC_PIXELS_PER_BYTE)
        } else {
            -((-delta) / MLX_IMAGE_COND_ALLOC_PIXELS_PER_BYTE)
        };
        let conditioning = i128::from(allocators.conditioning) + conditioning_growth;
        let denoise =
            i128::from(allocators.denoise) + MLX_IMAGE_DENOISE_ALLOC_PER_PIXEL_BYTES * delta;
        let decode = i128::from(allocators.decode) + MLX_IMAGE_DECODE_ALLOC_PER_PIXEL_BYTES * delta;
        Some(AnchorDerivedPhases {
            conditioning: widened_by(conditioning, MLX_IMAGE_ALLOCATOR_RESIDUAL_MARGIN)?,
            denoise: widened_by(denoise, MLX_IMAGE_ALLOCATOR_RESIDUAL_MARGIN)?,
            decode: widened_by(decode, MLX_IMAGE_ALLOCATOR_RESIDUAL_MARGIN)?,
        })
    }
}

/// The workload axes the MLX image derivation prices. There is no temporal axis, and no regime
/// flag at all: the anchor is the WIDEST (eager resident, unbounded) composition, so it
/// upper-bounds every composition the lane can execute — see
/// [`MemoryAnchor::derive_mlx_image_phase_peaks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorMlxImageDeriveRequest {
    pub width: u32,
    pub height: u32,
}

/// One priced cell from [`MemoryAnchorStore::derive_video_phase_peaks_for_cell`]: the derived
/// phases, the anchor that priced them, and the component-delta bytes the sibling fall-through
/// added. Zero means one of two things and the caller can tell them apart from `anchor`: the
/// exact-cell path took no sibling at all, or every axis this derivation crossed is priced at a
/// legitimate zero (see [`LTX25_VARIANT_DELTA_COMPONENTS`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchorCellDerivation<'a> {
    pub phases: AnchorDerivedPhases,
    pub anchor: &'a MemoryAnchor,
    pub delta_bytes: u64,
}

impl MemoryAnchorStore {
    /// The bound component delta for one `(model, lane, tier, axis, target value)`, or `None`
    /// when the shipped inventories price no such component — in which case the fall-through
    /// refuses that axis rather than inventing a size.
    pub fn component_delta_for(
        &self,
        model_id: &str,
        backend: AnchorBackend,
        tier: &str,
        axis: ComponentDeltaAxis,
        to: &str,
    ) -> Option<&ComponentDelta> {
        self.component_deltas.iter().find(|delta| {
            delta.model_id == model_id
                && delta.backend == backend
                && delta.tier == tier
                && delta.axis == axis
                && delta.to == to
        })
    }

    /// Price one `(model, lane, tier, transformer variant, decoder)` video cell: the exact anchor
    /// when the cell carries one, otherwise a SIBLING anchor of the same `(model, lane, tier)`
    /// plus the bound component deltas for every differing pipeline axis (epic 22505 E2,
    /// feature-end fix round — see the component-delta section comment for the conservative
    /// direction: deltas ADD the full shipped file size of components the target materializes and
    /// the sibling does not, and subtract nothing).
    ///
    /// The delta is added to every phase's RAW estimate before the shared
    /// [`ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`] widening: on the MLX lane the allocator retains a
    /// component's bytes across phase transitions, so no phase can be exempted from the add, and
    /// widening after the add keeps the delta under the same allocator-envelope pricing as the
    /// anchored terms.
    ///
    /// Sibling choice is deterministic: the fewest differing axes win, then the lexicographically
    /// smallest anchor id. A sibling differing on an axis with NO bound delta row is skipped — the
    /// caller keeps its floor, never a guessed size. That is distinct from a row bound at ZERO
    /// bytes, which is a priced crossing that happens to cost nothing and is taken normally: the
    /// distilled variant materializes no component its dev sibling lacks, and refusing that
    /// crossing would discard a derivation the evidence fully supports.
    pub fn derive_video_phase_peaks_for_cell(
        &self,
        model_id: &str,
        backend: AnchorBackend,
        tier: &str,
        transformer_variant: Ltx25TransformerVariant,
        decoder: Ltx25Decoder,
        request: AnchorDeriveRequest,
    ) -> Option<AnchorCellDerivation<'_>> {
        if let Some(anchor) = self.anchor_for(model_id, backend, tier, transformer_variant, decoder)
        {
            return Some(AnchorCellDerivation {
                phases: anchor.derive_video_phase_peaks(request)?,
                anchor,
                delta_bytes: 0,
            });
        }
        let mut candidates: Vec<(u32, &MemoryAnchor, u64)> = Vec::new();
        for anchor in self.anchors.iter().filter(|anchor| {
            anchor.model_id == model_id
                && anchor.backend == backend
                && anchor.tier == tier
                && anchor.transformer_variant.is_some()
                && anchor.decoder.is_some()
        }) {
            let mut differing = 0_u32;
            let mut delta_bytes = 0_u64;
            if anchor.transformer_variant != Some(transformer_variant) {
                differing += 1;
                let Some(delta) = self.component_delta_for(
                    model_id,
                    backend,
                    tier,
                    ComponentDeltaAxis::TransformerVariant,
                    transformer_variant_key(transformer_variant),
                ) else {
                    continue;
                };
                delta_bytes = delta_bytes.saturating_add(delta.bytes);
            }
            if anchor.decoder != Some(decoder) {
                differing += 1;
                let Some(delta) = self.component_delta_for(
                    model_id,
                    backend,
                    tier,
                    ComponentDeltaAxis::Decoder,
                    decoder_key(decoder),
                ) else {
                    continue;
                };
                delta_bytes = delta_bytes.saturating_add(delta.bytes);
            }
            candidates.push((differing, anchor, delta_bytes));
        }
        candidates.sort_by(|(left_axes, left, _), (right_axes, right, _)| {
            left_axes
                .cmp(right_axes)
                .then_with(|| left.id.cmp(&right.id))
        });
        for (_, sibling, delta_bytes) in candidates {
            let Some(raw) = sibling.derive_video_phase_estimates_raw(request) else {
                continue;
            };
            // The delta re-prices the SIBLING'S OWN measured intercepts (which embed its
            // component set); a phase priced by a cell-blind architecture bound takes no add —
            // the bound's provenance already upper-bounds every variant/decoder, and adding a
            // component the bound never held would only widen the estimate past what the
            // conservative rule requires.
            let priced = |estimate: RawPhaseEstimate| {
                if estimate.anchored {
                    estimate.bytes + i128::from(delta_bytes)
                } else {
                    estimate.bytes
                }
            };
            let Some(phases) = (|| {
                Some(AnchorDerivedPhases {
                    conditioning: widened(priced(raw.conditioning))?,
                    denoise: widened(priced(raw.denoise))?,
                    decode: widened(priced(raw.decode))?,
                })
            })() else {
                continue;
            };
            return Some(AnchorCellDerivation {
                phases,
                anchor: sibling,
                delta_bytes,
            });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

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
    ///
    /// sc-22512 (E8): the subject is FOUND, not required. A packaged store in which every anchor
    /// states its axes carries no axis-free row for this harness to interrogate — that is a corpus
    /// nobody measured that way, not a defect, so the test withholds its question instead of
    /// reddening. The lookup rule keeps full force on every axis-free row the store does carry.
    #[test]
    fn an_axis_free_anchor_answers_no_variant_keyed_lookup() {
        let Some(axis_free) = store()
            .anchors
            .iter()
            .find(|anchor| anchor.transformer_variant.is_none())
        else {
            return;
        };
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
    ///
    /// sc-22512 (E8): the bf16 distilled/conv cell is looked up, not required. A corpus that never
    /// measured that cell is absence — the request simply prices from the caller's floor — so the
    /// harness withholds its question rather than reddening. Every regime-handshake assertion below
    /// keeps full force whenever the cell IS measured.
    #[test]
    fn an_anchor_measured_in_a_bounded_regime_refuses_the_unbounded_request() {
        let Some(bounded) = store().anchor_for(
            "ltx_2_5",
            AnchorBackend::Mlx,
            "bf16",
            Ltx25TransformerVariant::Distilled,
            Ltx25Decoder::Conv,
        ) else {
            return;
        };
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
            let current = doctored["anchors"][ltx]["measuredRegime"][field]
                .as_bool()
                .unwrap_or_else(|| panic!("{field} is a declared regime flag"));
            doctored["anchors"][ltx]["measuredRegime"][field] = serde_json::json!(!current);
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
    // Currency: the model's own loader closure (sc-22511, E9).
    //
    // The DERIVATION of the key — what a pin bump, a sibling edit and an unreached shared-crate
    // edit do to it — is asked of the real pinned inference tree by
    // `scripts/anchor-loader-closure.test.mjs`, which is where the walk lives. What is asked here
    // is the COMPARISON: which anchors a rotated key stales, and which it leaves alone.
    // -------------------------------------------------------------------------------------

    fn packaged_closures() -> AnchorLoaderClosures {
        load_anchor_loader_closures(PACKAGED_ANCHOR_LOADER_CLOSURES)
            .expect("the packaged loader closures parse")
    }

    #[test]
    fn every_packaged_anchor_declares_a_loader_closure() {
        let store = load_memory_anchors(PACKAGED_MEMORY_ANCHORS).expect("packaged store loads");
        let closures = packaged_closures();
        assert!(!store.anchors.is_empty(), "the store must carry anchors");
        for anchor in &store.anchors {
            let key = anchor_loader_closure_key(&anchor.model_id, anchor.backend);
            assert!(
                closures.models.contains_key(&key),
                "anchor {} has no loader-closure declaration for {key}. Add the (model, lane) to \
                 config/anchor-loader-closures.json and derive it: node \
                 scripts/anchor-loader-closure.mjs --repo <inference clone> --write",
                anchor.id
            );
        }
    }

    /// CURRENCY IS REPORTED, NEVER ASSERTED — and that distinction is the point of E8.
    ///
    /// A packaged anchor whose model's loader source has moved since the measurement is STALE BY
    /// DESIGN: `is_current` returns false, admission demotes that cell to the conservative floor,
    /// and the render still runs. Asserting currency here would turn the first pin bump that
    /// genuinely touches a loader into a red `cargo test` on a change with nothing wrong in it —
    /// which is the pin-bump-forces-re-measurement gate this epic dismantled, rebuilt one level
    /// down. `bump-inference` regenerates the closures with `--write` automatically, so that red
    /// would land on the bump itself.
    ///
    /// What IS asserted is the loud half above: an anchor for a model nobody DECLARED is a
    /// mistake in the store, not a designed state, and it fails.
    #[test]
    fn packaged_anchor_currency_is_reported_not_gated() {
        let store = load_memory_anchors(PACKAGED_MEMORY_ANCHORS).expect("packaged store loads");
        let closures = packaged_closures();
        let stale: Vec<&str> = store
            .anchors
            .iter()
            .filter(|anchor| !anchor.is_current(&closures))
            .map(|anchor| anchor.id.as_str())
            .collect();
        if !stale.is_empty() {
            eprintln!(
                "note: {} of {} packaged anchors are not current against their model's declared \
                 loader closure and will demote to the conservative floor: {}",
                stale.len(),
                store.anchors.len(),
                stale.join(", ")
            );
        }
    }

    /// THE HEADLINE, comparison half: rotating one model's loader digest stales exactly that
    /// model's anchors — a sibling model declared beside it keeps its own.
    ///
    /// Both sides of this are REAL packaged anchors, not fabricated ones. `ltx_2_3` and `ltx_2_5`
    /// share the crate `mlx-gen-ltx`, which is precisely the pair the crate-level provider digest
    /// could not separate (E9's first named failure): under that unit a 2.3-only edit rotated 2.5.
    #[test]
    fn a_rotated_loader_digest_stales_exactly_that_models_anchors() {
        let store = load_memory_anchors(PACKAGED_MEMORY_ANCHORS).expect("packaged store loads");
        // A KNOWN-CURRENT BASELINE BY CONSTRUCTION: every declared digest is set to what that
        // model's own anchors recorded. This is a comparison test, and it must not silently become
        // a currency gate — whether the real pinned source still agrees is a separate question
        // whose answer is allowed to be "no" (a stale anchor demotes to the floor, by design).
        let mut closures = packaged_closures();
        for anchor in &store.anchors {
            let key = anchor_loader_closure_key(&anchor.model_id, anchor.backend);
            let declared = closures
                .models
                .get_mut(&key)
                .unwrap_or_else(|| panic!("{key} is declared"));
            declared
                .digest
                .clone_from(&anchor.source.loader_closure_digest);
        }
        let closures = closures;
        let subject = anchor_loader_closure_key("ltx_2_5", AnchorBackend::Mlx);
        let sibling = anchor_loader_closure_key("ltx_2_3", AnchorBackend::Mlx);
        assert_ne!(subject, sibling);

        let anchors_for = |key: &str| -> Vec<&MemoryAnchor> {
            store
                .anchors
                .iter()
                .filter(|anchor| anchor_loader_closure_key(&anchor.model_id, anchor.backend) == key)
                .collect()
        };
        let subject_anchors = anchors_for(&subject);
        let sibling_anchors = anchors_for(&sibling);
        assert!(
            !subject_anchors.is_empty() && !sibling_anchors.is_empty(),
            "both models must carry packaged anchors for this comparison to mean anything"
        );

        // Rotate ONE model's key at a time and read both populations back.
        let rotated = |key: &str, to: &str| {
            let mut moved = closures.clone();
            moved
                .models
                .get_mut(key)
                .unwrap_or_else(|| panic!("{key} is declared"))
                .digest = to.repeat(64);
            moved
        };

        // The sibling's loader moved: every ltx_2_5 anchor stays authoritative, every ltx_2_3
        // anchor stales.
        let moved_sibling = rotated(&sibling, "c");
        for anchor in &subject_anchors {
            assert!(
                anchor.is_current(&moved_sibling),
                "anchor {} must survive a sibling model's loader edit",
                anchor.id
            );
        }
        for anchor in &sibling_anchors {
            assert!(
                !anchor.is_current(&moved_sibling),
                "anchor {} must stale when ITS OWN loader moves",
                anchor.id
            );
        }

        // And the mirror image.
        let moved_subject = rotated(&subject, "d");
        for anchor in &subject_anchors {
            assert!(
                !anchor.is_current(&moved_subject),
                "anchor {} must stale when its OWN loader moves",
                anchor.id
            );
        }
        for anchor in &sibling_anchors {
            assert!(
                anchor.is_current(&moved_subject),
                "anchor {} must survive a sibling model's loader edit",
                anchor.id
            );
        }
    }

    /// The same anchor on the other backend lane is a different loader and a different key.
    #[test]
    fn currency_is_keyed_per_backend_lane_and_fails_closed_when_undeclared() {
        let store = load_memory_anchors(PACKAGED_MEMORY_ANCHORS).expect("packaged store loads");
        let closures = packaged_closures();
        let mut candle = store.anchors[0].clone();
        candle.backend = AnchorBackend::Candle;
        assert!(
            !candle.is_current(&closures),
            "an undeclared (model, lane) cannot be shown current, so it is not"
        );
    }

    /// The campaign fingerprint is provenance, not currency: a re-fingerprinted campaign over the
    /// same loader must not demote the anchor (E9).
    #[test]
    fn the_calibration_fingerprint_is_not_a_currency_term() {
        let store = load_memory_anchors(PACKAGED_MEMORY_ANCHORS).expect("packaged store loads");
        let anchor = &store.anchors[0];
        // Current BY CONSTRUCTION, for the reason spelled out in
        // `a_rotated_loader_digest_stales_exactly_that_models_anchors`: the claim here is about the
        // FINGERPRINT, and reading it off the live pinned source would quietly make it a currency
        // gate that reds on any pin bump the loader source actually moved through.
        let mut closures = packaged_closures();
        let key = anchor_loader_closure_key(&anchor.model_id, anchor.backend);
        closures
            .models
            .get_mut(&key)
            .unwrap_or_else(|| panic!("{key} is declared"))
            .digest
            .clone_from(&anchor.source.loader_closure_digest);

        let mut refingerprinted = anchor.clone();
        assert!(refingerprinted.is_current(&closures));
        refingerprinted.source.calibration_fingerprint = "sc-99999-some-later-campaign".to_owned();
        assert!(
            refingerprinted.is_current(&closures),
            "a later campaign fingerprint must not move currency"
        );
    }

    #[test]
    fn a_loader_closure_file_that_does_not_parse_or_declare_is_refused() {
        let mut version: serde_json::Value =
            serde_json::from_str(PACKAGED_ANCHOR_LOADER_CLOSURES).expect("closures parse");
        version["digestVersion"] = serde_json::json!("anchor-loader-closure v0");
        let error = load_anchor_loader_closures(&version.to_string())
            .expect_err("a version mismatch must be refused");
        assert!(error.contains("digest version"), "{error}");

        let mut digest: serde_json::Value =
            serde_json::from_str(PACKAGED_ANCHOR_LOADER_CLOSURES).expect("closures parse");
        digest["models"]["ltx_2_5:mlx"]["digest"] = serde_json::json!("not-a-digest");
        let error = load_anchor_loader_closures(&digest.to_string())
            .expect_err("a malformed digest must be refused");
        assert!(error.contains("is not a sha256"), "{error}");

        let mut rooted: serde_json::Value =
            serde_json::from_str(PACKAGED_ANCHOR_LOADER_CLOSURES).expect("closures parse");
        rooted["models"]["ltx_2_5:mlx"]["entryPoints"] =
            serde_json::json!(["crates/media/mlx-gen/mlx-gen-wan/src/model.rs"]);
        let error = load_anchor_loader_closures(&rooted.to_string())
            .expect_err("an entry point outside its own closure must be refused");
        assert!(error.contains("not in its own closure"), "{error}");

        let mut counted: serde_json::Value =
            serde_json::from_str(PACKAGED_ANCHOR_LOADER_CLOSURES).expect("closures parse");
        counted["models"]["ltx_2_5:mlx"]["closureFileCount"] = serde_json::json!(1);
        let error = load_anchor_loader_closures(&counted.to_string())
            .expect_err("a file count that disagrees with the list must be refused");
        assert!(error.contains("declares"), "{error}");
    }

    #[test]
    fn an_anchor_whose_loader_digest_is_not_a_sha256_is_refused_at_load() {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        doctored["anchors"][0]["source"]["loaderClosureDigest"] = serde_json::json!("nope");
        let error =
            load_memory_anchors(&doctored.to_string()).expect_err("the digest shape is checked");
        assert!(error.contains("loader closure digest"), "{error}");
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

    // -------------------------------------------------------------------------------------
    // MLX image lane (epic 22505 feature-end fix round, E2/E7).
    // -------------------------------------------------------------------------------------

    const IMAGE_MLX_CORPUS_PATH: &str = "docs/generated/memory-calibration-evidence.json";

    /// One retained MLX image record: geometry, composition, and its measured per-phase
    /// ALLOCATOR levels plus the overall envelope.
    struct MlxImageCorpusRecord {
        id: String,
        tier: String,
        width: u32,
        height: u32,
        load_shape: String,
        engaged: Vec<String>,
        conditioning_alloc: u64,
        denoise_alloc: u64,
        decode_alloc: u64,
        overall_envelope: u64,
    }

    fn mlx_image_corpus(model_id: &str) -> Vec<MlxImageCorpusRecord> {
        let raw = PACKAGED_MEMORY_ANCHOR_SOURCES
            .iter()
            .find(|(path, _)| *path == IMAGE_MLX_CORPUS_PATH)
            .map(|(_, raw)| *raw)
            .expect("the image-MLX retained corpus is compiled in");
        let source: serde_json::Value = serde_json::from_str(raw).expect("corpus parses");
        source["records"]
            .as_array()
            .expect("records")
            .iter()
            .filter(|record| {
                record["target"]["modelId"].as_str() == Some(model_id)
                    && record["backend"].as_str() == Some("mlx")
                    && record["target"]["geometry"]["frames"].as_u64() == Some(1)
            })
            .map(|record| {
                let geometry = &record["target"]["geometry"];
                let alloc = |phase: &str| {
                    record["observedMemory"][phase]["allocatorBytes"]
                        .as_u64()
                        .unwrap_or_else(|| panic!("{phase} allocatorBytes"))
                };
                MlxImageCorpusRecord {
                    id: record["id"].as_str().expect("id").to_owned(),
                    tier: record["target"]["tier"].as_str().expect("tier").to_owned(),
                    width: geometry["width"].as_u64().expect("width") as u32,
                    height: geometry["height"].as_u64().expect("height") as u32,
                    load_shape: record["loadShape"].as_str().expect("loadShape").to_owned(),
                    engaged: record["strategy"]["engagedRungs"]
                        .as_array()
                        .expect("engaged rungs")
                        .iter()
                        .filter_map(|rung| rung.as_str().map(str::to_owned))
                        .collect(),
                    conditioning_alloc: alloc("conditioning"),
                    denoise_alloc: alloc("denoise"),
                    decode_alloc: alloc("decode"),
                    overall_envelope: record["observedMemory"]["overall"]["allocatorBytes"]
                        .as_u64()
                        .expect("overall envelope"),
                }
            })
            .collect()
    }

    fn flux2_mlx_anchor(tier: &str) -> &'static MemoryAnchor {
        store()
            .image_anchor_for("flux2_dev", AnchorBackend::Mlx, tier)
            .unwrap_or_else(|| panic!("the flux2_dev {tier} MLX anchor is packaged"))
    }

    /// The per-model scope of the image-MLX law, read off the packaged store: the models whose own
    /// retained records vary geometry (flux2_dev, both tiers) derive; the single-geometry models
    /// (qwen_image, z_image_turbo) and the axis-free video anchor (ltx_2_3) are validation-only
    /// with their reason stated, and the law honors the statement.
    #[test]
    fn image_mlx_derivability_is_scoped_per_model_with_stated_reasons() {
        for tier in ["q4", "q8"] {
            let anchor = flux2_mlx_anchor(tier);
            assert_eq!(anchor.underived_reason, None, "flux2 {tier} must derive");
            assert!(anchor
                .derive_mlx_image_phase_peaks(AnchorMlxImageDeriveRequest {
                    width: 1024,
                    height: 1024,
                })
                .is_some());
        }
        for (model, tiers) in [
            ("qwen_image", &["q4", "q8", "bf16"][..]),
            ("z_image_turbo", &["q4"][..]),
        ] {
            for tier in tiers {
                let anchor = store()
                    .image_anchor_for(model, AnchorBackend::Mlx, tier)
                    .unwrap_or_else(|| panic!("{model} {tier} anchor"));
                let reason = anchor
                    .underived_reason
                    .as_deref()
                    .unwrap_or_else(|| panic!("{model} {tier} must state why it is underived"));
                assert!(
                    reason.contains("single geometry"),
                    "{model} {tier}: the reason must name the missing spread, got {reason}"
                );
                assert!(
                    anchor
                        .derive_mlx_image_phase_peaks(AnchorMlxImageDeriveRequest {
                            width: anchor.geometry.width,
                            height: anchor.geometry.height,
                        })
                        .is_none(),
                    "{model} {tier}: an underived anchor must price nothing"
                );
            }
        }
        let ltx23 = store()
            .anchors
            .iter()
            .find(|anchor| anchor.model_id == "ltx_2_3")
            .expect("the ltx_2_3 anchor is packaged");
        assert!(
            ltx23
                .underived_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("pipeline axes")),
            "ltx_2_3's axis-free anchor must state its refusal"
        );
    }

    /// AC (E2/E7): the image-MLX derivation brackets every retained record of the derivable
    /// models — per-phase against the measured ALLOCATOR levels, and overall against the measured
    /// admission envelope, inside the named tightness budget.
    #[test]
    fn mlx_image_derivation_brackets_every_retained_flux2_measurement() {
        let corpus = mlx_image_corpus("flux2_dev");
        // Shape: both anchored tiers and both measured geometries must appear, or the test stops
        // asking its question.
        for tier in ["q4", "q8"] {
            for edge in [768_u32, 1024] {
                assert!(
                    corpus
                        .iter()
                        .any(|record| record.tier == tier && record.width == edge),
                    "the retained corpus must cover {tier} at {edge}x{edge}"
                );
            }
        }
        let mut bracketed = 0_usize;
        for record in &corpus {
            let anchor = flux2_mlx_anchor(&record.tier);
            let derived = anchor
                .derive_mlx_image_phase_peaks(AnchorMlxImageDeriveRequest {
                    width: record.width,
                    height: record.height,
                })
                .unwrap_or_else(|| panic!("{} must be derivable", record.id));
            for (phase, derived_bytes, measured_bytes) in [
                (
                    "conditioning",
                    derived.conditioning,
                    record.conditioning_alloc,
                ),
                ("denoise", derived.denoise, record.denoise_alloc),
                ("decode", derived.decode, record.decode_alloc),
            ] {
                assert!(
                    derived_bytes >= measured_bytes,
                    "{} {phase}: derived allocator level {derived_bytes} under the measured \
                     {measured_bytes}",
                    record.id
                );
            }
            let upper = derived.peak_bytes();
            assert!(
                upper >= record.overall_envelope,
                "{}: derived admission bound {upper} under the measured envelope {}",
                record.id,
                record.overall_envelope
            );
            let tight_cap = ((record.overall_envelope as f64)
                * (1.0 + MLX_IMAGE_VALIDATION_TIGHTNESS_BUDGET))
                .ceil() as u64;
            assert!(
                upper <= tight_cap,
                "{}: derived admission bound {upper} exceeds the tightness cap {tight_cap}",
                record.id
            );
            bracketed += 1;
        }
        assert!(
            bracketed >= 8,
            "both tiers at both geometries with their repeat captures must be bracketed, got \
             {bracketed}"
        );
    }

    /// The image-MLX slopes are pinned from both sides: at or ABOVE the highest within-cell
    /// measured allocator slope (upward extrapolation must not cross below the trend), and BELOW
    /// the ceiling at which the margin-widened 768x768 downward derivation falls under the
    /// measured 768x768 allocator level.
    #[test]
    fn every_mlx_image_coefficient_sits_inside_the_window_the_retained_pairs_allow() {
        let corpus = mlx_image_corpus("flux2_dev");
        let span = 1024_i128 * 1024 - MLX_IMAGE_SMALLEST_RETAINED_PIXELS;
        let mut cond_slopes: Vec<f64> = Vec::new();
        let mut denoise_slopes: Vec<f64> = Vec::new();
        let mut decode_slopes: Vec<f64> = Vec::new();
        for (index, left) in corpus.iter().enumerate() {
            for right in &corpus[index + 1..] {
                if left.tier != right.tier
                    || left.load_shape != right.load_shape
                    || left.engaged != right.engaged
                {
                    continue;
                }
                let left_px = i128::from(left.width) * i128::from(left.height);
                let right_px = i128::from(right.width) * i128::from(right.height);
                if left_px == right_px {
                    continue;
                }
                let delta = (right_px - left_px) as f64;
                cond_slopes.push(
                    (right.conditioning_alloc as f64 - left.conditioning_alloc as f64) / delta,
                );
                denoise_slopes
                    .push((right.denoise_alloc as f64 - left.denoise_alloc as f64) / delta);
                decode_slopes.push((right.decode_alloc as f64 - left.decode_alloc as f64) / delta);
            }
        }
        assert!(
            !denoise_slopes.is_empty(),
            "the corpus must measure within-cell slopes"
        );
        let highest = |slopes: &[f64]| slopes.iter().fold(f64::MIN, |a, &b| a.max(b.abs()));
        for tier in ["q4", "q8"] {
            let anchor = flux2_mlx_anchor(tier);
            let allocators = anchor
                .phase_allocator_envelope_bytes
                .expect("flux2 anchors carry phase allocator levels");
            let measured_768 = corpus
                .iter()
                .find(|record| record.tier == tier && record.width == 768)
                .expect("the 768x768 retained row");
            for (phase, coefficient, intercept, small_measured, slopes) in [
                (
                    "conditioning",
                    1.0 / MLX_IMAGE_COND_ALLOC_PIXELS_PER_BYTE as f64,
                    allocators.conditioning,
                    measured_768.conditioning_alloc,
                    &cond_slopes,
                ),
                (
                    "denoise",
                    MLX_IMAGE_DENOISE_ALLOC_PER_PIXEL_BYTES as f64,
                    allocators.denoise,
                    measured_768.denoise_alloc,
                    &denoise_slopes,
                ),
                (
                    "decode",
                    MLX_IMAGE_DECODE_ALLOC_PER_PIXEL_BYTES as f64,
                    allocators.decode,
                    measured_768.decode_alloc,
                    &decode_slopes,
                ),
            ] {
                let observed = highest(slopes);
                assert!(
                    coefficient >= observed,
                    "{phase}: coefficient {coefficient} sits below the highest measured \
                     within-cell allocator slope {observed}"
                );
                let ceiling = (intercept as f64
                    - small_measured as f64 / (1.0 + MLX_IMAGE_ALLOCATOR_RESIDUAL_MARGIN))
                    / span as f64;
                assert!(
                    coefficient <= ceiling,
                    "{tier} {phase}: coefficient {coefficient} exceeds the {ceiling} ceiling at \
                     which the margin-widened 768x768 derivation falls under the measured level"
                );
            }
        }
    }

    /// The identity and regime guards of the image-MLX law, each mutated individually on an
    /// otherwise-derivable anchor.
    #[test]
    fn the_mlx_image_law_refuses_foreign_identities_and_regimes() {
        let request = AnchorMlxImageDeriveRequest {
            width: 1024,
            height: 1024,
        };
        assert!(
            flux2_mlx_anchor("q4")
                .derive_mlx_image_phase_peaks(request)
                .is_some(),
            "the unmutated anchor must derive, or the mutations below prove nothing"
        );
        for (label, mutate) in [
            (
                "the candle lane",
                Box::new(|anchor: &mut MemoryAnchor| anchor.backend = AnchorBackend::Candle)
                    as Box<dyn Fn(&mut MemoryAnchor)>,
            ),
            (
                "a video pipeline variant",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.transformer_variant = Some(Ltx25TransformerVariant::Dev);
                }),
            ),
            (
                "a video decoder",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.decoder = Some(Ltx25Decoder::Conv);
                }),
            ),
            (
                "a multi-frame measured geometry",
                Box::new(|anchor: &mut MemoryAnchor| anchor.geometry.frames = 145),
            ),
            (
                "deferred materialization",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.load_shape = AnchorLoadShape::DeferredMaterialization;
                }),
            ),
            (
                "a staged measured regime",
                Box::new(|anchor: &mut MemoryAnchor| anchor.measured_regime.staged = true),
            ),
            (
                "a tiled measured regime",
                Box::new(|anchor: &mut MemoryAnchor| anchor.measured_regime.decode_tiled = true),
            ),
            (
                "a chunked-attention measured regime",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.measured_regime.attention_chunked = true;
                }),
            ),
            (
                "a windowed measured regime",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.measured_regime.transformer_windowed = true;
                }),
            ),
            (
                "no phase allocator decomposition",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.phase_allocator_envelope_bytes = None;
                }),
            ),
            (
                "a stated underived reason",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.underived_reason = Some("validation-only".to_owned());
                }),
            ),
        ] {
            let mut mutated = flux2_mlx_anchor("q4").clone();
            mutate(&mut mutated);
            assert!(
                mutated.derive_mlx_image_phase_peaks(request).is_none(),
                "the MLX image law must refuse an anchor carrying {label}"
            );
        }
        // And the video/candle laws refuse the MLX image anchor in turn.
        assert!(flux2_mlx_anchor("q4")
            .derive_video_phase_peaks(plain_request(1024, 1024, 1))
            .is_none());
        assert!(flux2_mlx_anchor("q4")
            .derive_image_phase_peaks(AnchorImageDeriveRequest {
                width: 1024,
                height: 1024,
                staged_residency: true,
            })
            .is_none());
    }

    #[test]
    fn mlx_image_derived_peaks_track_output_area_and_clamp_below_the_corpus() {
        let anchor = flux2_mlx_anchor("q8");
        let at = |width: u32, height: u32| {
            anchor
                .derive_mlx_image_phase_peaks(AnchorMlxImageDeriveRequest { width, height })
                .unwrap_or_else(|| panic!("{width}x{height} must be derivable"))
        };
        assert!(at(896, 896).peak_bytes() < at(1024, 1024).peak_bytes());
        assert!(at(1024, 1024).peak_bytes() < at(1536, 1536).peak_bytes());
        // Area, not aspect.
        assert_eq!(at(1344, 768).peak_bytes(), at(768, 1344).peak_bytes());
        // Sub-corpus requests clamp to the smallest retained geometry.
        let floor = at(768, 768);
        for (label, clamped) in [("1x1", at(1, 1)), ("512x512", at(512, 512))] {
            assert_eq!(
                clamped, floor,
                "{label} must be priced at the 768x768 clamp floor"
            );
        }
        assert!(anchor
            .derive_mlx_image_phase_peaks(AnchorMlxImageDeriveRequest {
                width: 0,
                height: 512,
            })
            .is_none());
    }

    /// An underived anchor prices nothing on ANY lane — the store's statement and the runtime
    /// behaviour cannot disagree.
    #[test]
    fn an_underived_anchor_prices_nothing_on_any_lane() {
        let mut candle = krea_candle_anchor().clone();
        candle.underived_reason = Some("validation-only".to_owned());
        assert!(candle
            .derive_image_phase_peaks(AnchorImageDeriveRequest {
                width: 1024,
                height: 1024,
                staged_residency: true,
            })
            .is_none());
        let mut video = store()
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q8",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae,
            )
            .expect("q8 dev/diffvae anchor")
            .clone();
        video.underived_reason = Some("validation-only".to_owned());
        assert!(video
            .derive_video_phase_peaks(plain_request(768, 512, 121))
            .is_none());
    }

    // -------------------------------------------------------------------------------------
    // Variant/decoder component deltas (epic 22505 E2, feature-end fix round).
    // -------------------------------------------------------------------------------------

    /// E2's headline: an unmeasured (variant, decoder) cell of an anchored (model, tier, lane)
    /// derives from the sibling anchor plus the bound file-size delta — and the delta is APPLIED,
    /// byte for byte, so zeroing it reds this test.
    ///
    /// The crossing exercised is DEV-from-distilled, because that is the direction the corrected
    /// table prices above zero (the dev recipe materializes the distillation LoRA the distilled
    /// checkpoint already contains — see [`LTX25_VARIANT_DELTA_COMPONENTS`]). The mirrored
    /// distilled-from-dev crossing is a legitimate zero and is asserted by
    /// `a_zero_crossing_derives_the_sibling_estimate_unchanged_rather_than_refusing`.
    #[test]
    fn an_unmeasured_variant_cell_derives_from_the_sibling_anchor_plus_the_bound_delta() {
        // Removing the bf16 dev anchors makes bf16 dev/diffvae an unmeasured cell whose only
        // one-axis sibling is bf16 distilled/diffvae — i.e. exactly the DEV-from-distilled variant
        // crossing, with no decoder crossing mixed in. The request runs the deferred and windowed
        // rungs the retained distilled measurement ran (an anchor never prices a request looser
        // than its own regime) and leaves decode untiled, so the decode phase is priced from that
        // anchor's measured intercept and the add has somewhere to land.
        let request = AnchorDeriveRequest {
            transformer_windowed: true,
            deferred_materialization: true,
            ..plain_request(1280, 704, 145)
        };
        let mut doctored = store().clone();
        doctored.anchors.retain(|anchor| {
            !(anchor.model_id == "ltx_2_5"
                && anchor.tier == "bf16"
                && anchor.transformer_variant == Some(Ltx25TransformerVariant::Dev))
        });
        assert!(doctored
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "bf16",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae
            )
            .is_none());
        let derived = doctored
            .derive_video_phase_peaks_for_cell(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "bf16",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae,
                request,
            )
            .expect("the unmeasured variant cell derives from its sibling");
        let sibling = doctored
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "bf16",
                Ltx25TransformerVariant::Distilled,
                Ltx25Decoder::DiffVae,
            )
            .expect("the sibling anchor");
        assert_eq!(derived.anchor.id, sibling.id);
        let delta = store()
            .component_delta_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "bf16",
                ComponentDeltaAxis::TransformerVariant,
                "dev",
            )
            .expect("the dev variant delta is bound");
        assert_eq!(derived.delta_bytes, delta.bytes);
        // The delta names the distillation LoRA the DEV recipe materializes on top of the shared
        // transformer, and its bytes are the shipped file's size — not a tuned number.
        assert_eq!(
            delta.files,
            vec!["distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors".to_owned()]
        );
        assert!(delta.bytes > 0);
        let raw = sibling
            .derive_video_phase_estimates_raw(request)
            .expect("the sibling prices the request");
        let expect = |estimate: RawPhaseEstimate| {
            if estimate.anchored {
                widened(estimate.bytes + i128::from(delta.bytes)).expect("widens")
            } else {
                widened(estimate.bytes).expect("widens")
            }
        };
        assert_eq!(derived.phases.conditioning, expect(raw.conditioning));
        assert_eq!(derived.phases.denoise, expect(raw.denoise));
        assert_eq!(derived.phases.decode, expect(raw.decode));
        // The conservative direction: the delta ADDS — every intercept-priced phase of the derived
        // cell sits strictly above the sibling's own derivation.
        let plain = sibling
            .derive_video_phase_peaks(request)
            .expect("the sibling's own derivation");
        let mut lifted = 0_usize;
        for (anchored, derived_bytes, plain_bytes) in [
            (
                raw.conditioning.anchored,
                derived.phases.conditioning,
                plain.conditioning,
            ),
            (raw.denoise.anchored, derived.phases.denoise, plain.denoise),
            (raw.decode.anchored, derived.phases.decode, plain.decode),
        ] {
            if anchored {
                assert!(derived_bytes > plain_bytes);
                lifted += 1;
            } else {
                assert_eq!(derived_bytes, plain_bytes);
            }
        }
        assert!(lifted > 0, "the crossing must lift at least one phase");
    }

    /// The mirrored crossing: DISTILLED-from-dev crosses no component, so the fall-through must
    /// return the sibling's own estimate unchanged — NOT refuse the cell.
    ///
    /// This is the whole point of representing a zero delta instead of rejecting one. Before the
    /// feature-end correction the distilled row carried the dev recipe's LoRA, which made this
    /// cell's bound 8.9 GB wider than the evidence entails while the mirrored dev cell was 8.9 GB
    /// too narrow — the OOM direction.
    #[test]
    fn a_zero_crossing_derives_the_sibling_estimate_unchanged_rather_than_refusing() {
        // q8 distilled/diffvae carries no anchor; q8 dev/diffvae is its variant sibling.
        assert!(store()
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q8",
                Ltx25TransformerVariant::Distilled,
                Ltx25Decoder::DiffVae
            )
            .is_none());
        let request = plain_request(1280, 704, 145);
        let derived = store()
            .derive_video_phase_peaks_for_cell(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q8",
                Ltx25TransformerVariant::Distilled,
                Ltx25Decoder::DiffVae,
                request,
            )
            .expect("a zero crossing still derives — a zero delta is priced, not missing");
        let sibling = store()
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q8",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae,
            )
            .expect("the sibling anchor");
        assert_eq!(derived.anchor.id, sibling.id);
        let delta = store()
            .component_delta_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q8",
                ComponentDeltaAxis::TransformerVariant,
                "distilled",
            )
            .expect("the distilled variant delta is bound — as an explicit zero");
        assert_eq!(delta.bytes, 0);
        assert!(delta.files.is_empty());
        assert_eq!(derived.delta_bytes, 0);
        let plain = sibling
            .derive_video_phase_peaks(request)
            .expect("the sibling's own derivation");
        assert_eq!(derived.phases.conditioning, plain.conditioning);
        assert_eq!(derived.phases.denoise, plain.denoise);
        assert_eq!(derived.phases.decode, plain.decode);
    }

    /// The fall-through prefers the exact anchor, then the fewest differing axes, then the
    /// smallest anchor id — deterministically.
    #[test]
    fn the_delta_fall_through_prefers_the_exact_anchor_then_the_fewest_axes() {
        let request = plain_request(1280, 704, 145);
        let exact = store()
            .derive_video_phase_peaks_for_cell(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "q8",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae,
                request,
            )
            .expect("the exact cell derives");
        assert_eq!(exact.delta_bytes, 0);
        assert_eq!(
            exact.anchor.transformer_variant,
            Some(Ltx25TransformerVariant::Dev)
        );
        // Remove the bf16 distilled/diffvae anchor: bf16 distilled/diffvae then has TWO one-axis
        // siblings (dev/diffvae by variant, distilled/conv by decoder) and the smaller anchor id
        // (dev/diffvae) must win, with the variant delta.
        let mut doctored = store().clone();
        doctored.anchors.retain(|anchor| {
            !(anchor.model_id == "ltx_2_5"
                && anchor.tier == "bf16"
                && anchor.transformer_variant == Some(Ltx25TransformerVariant::Distilled)
                && anchor.decoder == Some(Ltx25Decoder::DiffVae))
        });
        let derived = doctored
            .derive_video_phase_peaks_for_cell(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "bf16",
                Ltx25TransformerVariant::Distilled,
                Ltx25Decoder::DiffVae,
                request,
            )
            .expect("the doctored store still derives the cell via a sibling");
        assert_eq!(
            derived.anchor.transformer_variant,
            Some(Ltx25TransformerVariant::Dev)
        );
        assert_eq!(derived.anchor.decoder, Some(Ltx25Decoder::DiffVae));
        // A cell with NO priced sibling axis fails to the caller's floor: strip the deltas and
        // the same lookup returns nothing.
        let mut unpriced = doctored.clone();
        unpriced.component_deltas.clear();
        assert!(
            unpriced
                .derive_video_phase_peaks_for_cell(
                    "ltx_2_5",
                    AnchorBackend::Mlx,
                    "bf16",
                    Ltx25TransformerVariant::Distilled,
                    Ltx25Decoder::DiffVae,
                    request,
                )
                .is_none(),
            "an axis with no bound delta must fall to the floor, never a guessed size"
        );
    }

    /// Leave-one-out validation against the retained records that DO exist for cells the
    /// fall-through can reach: with the bf16 distilled anchors removed, every retained bf16
    /// distilled record must be bracketed by the sibling-plus-delta derivation within the delta
    /// budget — diffvae via the variant crossing (a priced ZERO under the corrected table, since
    /// distilled materializes nothing its dev sibling lacks), conv via that plus the decoder
    /// crossing.
    ///
    /// The mirrored direction — dev records derived from distilled siblings, which crosses the
    /// 8.9 GB LoRA — is NOT reachable from the retained corpus and is deliberately not faked here:
    /// every retained bf16 distilled measurement engages `bounded_transformer_residency` while
    /// neither bf16 dev record does, and `derive_video_phase_estimates_raw` refuses to price a
    /// non-windowed request from a windowed anchor. The direction is covered instead by
    /// `an_unmeasured_variant_cell_derives_from_the_sibling_anchor_plus_the_bound_delta` (byte-for
    /// -byte application) and by the engine-agreement tests (which key the table itself).
    #[test]
    fn leave_one_out_sibling_delta_derivations_bracket_the_retained_distilled_records() {
        let mut doctored = store().clone();
        doctored.anchors.retain(|anchor| {
            !(anchor.model_id == "ltx_2_5"
                && anchor.transformer_variant == Some(Ltx25TransformerVariant::Distilled))
        });
        let corpus = retained_corpus();
        let distilled: Vec<_> = corpus
            .iter()
            .filter(|record| record.transformer_variant == Ltx25TransformerVariant::Distilled)
            .collect();
        assert!(
            distilled.len() >= 4,
            "the retained corpus must carry distilled records to validate against, got {}",
            distilled.len()
        );
        assert!(
            distilled
                .iter()
                .any(|record| record.decoder == Ltx25Decoder::Conv),
            "the distilled/conv records are part of the validation set"
        );
        let mut crossed_a_nonzero_delta = false;
        for record in distilled {
            let derived = doctored
                .derive_video_phase_peaks_for_cell(
                    "ltx_2_5",
                    AnchorBackend::Mlx,
                    &record.tier,
                    record.transformer_variant,
                    record.decoder,
                    AnchorDeriveRequest {
                        width: record.width,
                        height: record.height,
                        frames: record.frames,
                        decode_tiled: record.decode_tiled,
                        transformer_windowed: record.transformer_windowed,
                        deferred_materialization: record.deferred,
                    },
                )
                .unwrap_or_else(|| panic!("{} derives via a sibling", record.id));
            assert_ne!(
                derived.anchor.transformer_variant,
                Some(Ltx25TransformerVariant::Distilled),
                "{}: the leave-one-out derivation must come from the surviving variant",
                record.id
            );
            // The delta the fall-through applied must be EXACTLY what the shipped table prices for
            // this crossing — never an invented add, never a dropped one. Zero is a real answer
            // here, not a skipped axis: the distilled target crosses no component its dev sibling
            // lacks, so the variant term contributes nothing and only a decoder crossing lifts it.
            let mut expected = 0_u64;
            if derived.anchor.transformer_variant != Some(record.transformer_variant) {
                expected += doctored
                    .component_delta_for(
                        "ltx_2_5",
                        AnchorBackend::Mlx,
                        &record.tier,
                        ComponentDeltaAxis::TransformerVariant,
                        transformer_variant_key(record.transformer_variant),
                    )
                    .unwrap_or_else(|| panic!("{}: the variant crossing is priced", record.id))
                    .bytes;
            }
            if derived.anchor.decoder != Some(record.decoder) {
                expected += doctored
                    .component_delta_for(
                        "ltx_2_5",
                        AnchorBackend::Mlx,
                        &record.tier,
                        ComponentDeltaAxis::Decoder,
                        decoder_key(record.decoder),
                    )
                    .unwrap_or_else(|| panic!("{}: the decoder crossing is priced", record.id))
                    .bytes;
            }
            assert_eq!(
                derived.delta_bytes, expected,
                "{}: the applied delta must equal the shipped table's price for the crossing",
                record.id
            );
            crossed_a_nonzero_delta |= derived.delta_bytes > 0;
            // Where the derivation reuses a measured intercept, a NONZERO crossed-axis delta must
            // be APPLIED to it — a zeroed delta reds here — and a ZERO one must leave it exactly
            // alone. Architecture-bounded phases take no add (the bounds are cell-blind by their
            // documented provenance), so the check names the intercept-priced phases of this
            // record's regime.
            let sibling_raw = derived
                .anchor
                .derive_video_phase_estimates_raw(AnchorDeriveRequest {
                    width: record.width,
                    height: record.height,
                    frames: record.frames,
                    decode_tiled: record.decode_tiled,
                    transformer_windowed: record.transformer_windowed,
                    deferred_materialization: record.deferred,
                })
                .expect("the sibling prices the request");
            for (phase, anchored, sibling_raw, derived_bytes) in [
                (
                    "conditioning",
                    sibling_raw.conditioning.anchored,
                    sibling_raw.conditioning.bytes,
                    derived.phases.conditioning,
                ),
                (
                    "denoise",
                    sibling_raw.denoise.anchored,
                    sibling_raw.denoise.bytes,
                    derived.phases.denoise,
                ),
                (
                    "decode",
                    sibling_raw.decode.anchored,
                    sibling_raw.decode.bytes,
                    derived.phases.decode,
                ),
            ] {
                if anchored {
                    let undelta = widened(sibling_raw).expect("widens");
                    if derived.delta_bytes > 0 {
                        assert!(
                            derived_bytes > undelta,
                            "{} {phase}: the crossed-axis delta must be applied to the sibling's \
                             measured intercept",
                            record.id
                        );
                    } else {
                        assert_eq!(
                            derived_bytes, undelta,
                            "{} {phase}: a zero crossing must add nothing to the sibling's \
                             measured intercept",
                            record.id
                        );
                    }
                }
            }
            for (phase, derived_bytes, measured_bytes) in [
                (
                    "conditioning",
                    derived.phases.conditioning,
                    record.conditioning_active,
                ),
                ("denoise", derived.phases.denoise, record.denoise_active),
                ("decode", derived.phases.decode, record.decode_active),
            ] {
                assert!(
                    derived_bytes >= measured_bytes,
                    "{} {phase}: sibling+delta derived {derived_bytes} under measured \
                     {measured_bytes}",
                    record.id
                );
            }
            let upper = derived.phases.peak_bytes();
            assert!(
                upper >= record.overall_envelope,
                "{}: derived bound {upper} under the measured envelope {}",
                record.id,
                record.overall_envelope
            );
            let cap = ((record.overall_envelope as f64)
                * (1.0 + ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET))
                .ceil() as u64;
            assert!(
                upper <= cap,
                "{}: derived bound {upper} exceeds the delta tightness cap {cap} over envelope {} \
                 (overshoot {:.4})",
                record.id,
                record.overall_envelope,
                (upper as f64) / (record.overall_envelope as f64) - 1.0
            );
        }
        // The budget is a claim about DELTA-priced bounds, so the leave-one-out set has to carry
        // at least one crossing the table prices above zero. Without this the suite could pass on
        // zero crossings alone and never interrogate the add at all.
        assert!(
            crossed_a_nonzero_delta,
            "the leave-one-out set must include a crossing the delta table prices above zero"
        );
    }

    /// The delta budget is DERIVED, not chosen: it is the smallest round figure that clears every
    /// leave-one-out overshoot the corrected table produces, and it must stay meaningfully tight —
    /// a budget far above the binding case would stop refusing runaway estimates, which is the
    /// only job it has. Recomputed at the feature-end fix round, after the variant deltas were
    /// re-keyed onto the engine's real materialization.
    #[test]
    fn the_delta_tightness_budget_is_the_binding_leave_one_out_overshoot_rounded_up() {
        let mut doctored = store().clone();
        doctored.anchors.retain(|anchor| {
            !(anchor.model_id == "ltx_2_5"
                && anchor.transformer_variant == Some(Ltx25TransformerVariant::Distilled))
        });
        let mut worst = 0.0_f64;
        let mut worst_id = String::new();
        for record in retained_corpus()
            .iter()
            .filter(|record| record.transformer_variant == Ltx25TransformerVariant::Distilled)
        {
            let Some(derived) = doctored.derive_video_phase_peaks_for_cell(
                "ltx_2_5",
                AnchorBackend::Mlx,
                &record.tier,
                record.transformer_variant,
                record.decoder,
                AnchorDeriveRequest {
                    width: record.width,
                    height: record.height,
                    frames: record.frames,
                    decode_tiled: record.decode_tiled,
                    transformer_windowed: record.transformer_windowed,
                    deferred_materialization: record.deferred,
                },
            ) else {
                continue;
            };
            let overshoot =
                (derived.phases.peak_bytes() as f64) / (record.overall_envelope as f64) - 1.0;
            if overshoot > worst {
                worst = overshoot;
                worst_id = record.id.clone();
            }
        }
        assert!(
            worst > 0.0,
            "the leave-one-out set must produce a measurable overshoot to size the budget from"
        );
        assert!(
            ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET >= worst,
            "the budget {ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET} is under the binding \
             leave-one-out overshoot {worst:.4} ({worst_id})"
        );
        // Tight: no more than one percentage point of round-up slack above the binding case. This
        // is what stops the budget from quietly re-widening the next time a delta moves.
        assert!(
            ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET <= worst + 0.01,
            "the budget {ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET} is more than a point above the \
             binding leave-one-out overshoot {worst:.4} ({worst_id}) — re-derive it"
        );
    }

    /// The shipped variant-delta rows are keyed on what each variant MATERIALIZES.
    ///
    /// This is the test the first cut of the table did not have, and its absence is why the
    /// mapping shipped inverted: the delta rows were keyed on the variant's NAME (the "distilled"
    /// row got the distillation LoRA) rather than on the engine's load spec, which adds that LoRA
    /// for DEV. Re-inverting [`LTX25_VARIANT_DELTA_COMPONENTS`] or the extractor's `variantRows`
    /// reds here.
    #[test]
    fn ltx25_variant_delta_matches_the_engine_materialization() {
        let expected: BTreeMap<&str, BTreeSet<&str>> = LTX25_VARIANT_DELTA_COMPONENTS
            .iter()
            .map(|(variant, components)| (*variant, components.iter().copied().collect()))
            .collect();
        let mut seen: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        let mut tiers: BTreeSet<&str> = BTreeSet::new();
        for delta in store()
            .component_deltas
            .iter()
            .filter(|delta| delta.model_id == "ltx_2_5")
            .filter(|delta| delta.axis == ComponentDeltaAxis::TransformerVariant)
        {
            tiers.insert(delta.tier.as_str());
            // The component a row crosses is its file's top-level directory in the bundle.
            let components: BTreeSet<&str> = delta
                .files
                .iter()
                .map(|file| {
                    file.split('/')
                        .next()
                        .expect("an inventory path has a leading segment")
                })
                .collect();
            let previous = seen.insert(delta.to.as_str(), components.clone());
            if let Some(previous) = previous {
                assert_eq!(
                    previous, components,
                    "{}: every tier must cross the same components for a variant",
                    delta.id
                );
            }
        }
        assert_eq!(
            seen, expected,
            "the shipped variant deltas must cross exactly what the engine materializes"
        );
        // Every tier carries a row for BOTH variants — including the zero one, so the fall-through
        // can cross the variant axis in either direction rather than refusing half of it.
        assert!(!tiers.is_empty(), "the store must price the variant axis");
        for tier in &tiers {
            for (variant, _) in LTX25_VARIANT_DELTA_COMPONENTS {
                assert!(
                    store()
                        .component_delta_for(
                            "ltx_2_5",
                            AnchorBackend::Mlx,
                            tier,
                            ComponentDeltaAxis::TransformerVariant,
                            variant,
                        )
                        .is_some(),
                    "{tier}/{variant}: a missing row means \"unpriced axis\", which is a \
                     different claim from a zero crossing"
                );
            }
        }
    }

    /// The cross-check behind [`LTX25_VARIANT_DELTA_COMPONENTS`]: the mirror must still describe
    /// what the two engine sources actually do. `sceneworks-core` cannot depend on the worker or
    /// the capture arm, so the agreement is asserted against their SOURCE TEXT — an engine edit
    /// that moves the adapter to the other variant, or makes the enhancer variant-conditional,
    /// reds here instead of silently invalidating every variant delta.
    #[test]
    fn ltx25_variant_delta_components_mirror_the_engine_source() {
        const CAPTURE_ARM: &str =
            include_str!("../../sceneworks-memory-adapter/src/bin/mlx_ltx25.rs");
        const WORKER: &str = include_str!("../../sceneworks-worker/src/video_jobs/ltx.rs");

        // 1. The capture arm's adapter constant names the component the dev row prices.
        let adapter_decl = CAPTURE_ARM
            .lines()
            .find(|line| line.trim_start().starts_with("const DEV_ADAPTER: &str ="))
            .expect("the capture arm declares DEV_ADAPTER");
        let adapter_path = adapter_decl
            .split('"')
            .nth(1)
            .expect("DEV_ADAPTER is a string literal");
        let adapter_component = adapter_path
            .split('/')
            .next()
            .expect("DEV_ADAPTER is a bundle-relative path");
        let dev_components: BTreeSet<&str> = LTX25_VARIANT_DELTA_COMPONENTS
            .iter()
            .find(|(variant, _)| *variant == "dev")
            .map(|(_, components)| components.iter().copied().collect())
            .expect("the mirror describes dev");
        assert_eq!(
            dev_components,
            BTreeSet::from([adapter_component]),
            "the dev row must cross exactly the component DEV_ADAPTER names"
        );

        // 2. That adapter is pushed under the DEV arm — the nearest variant test above the push.
        let push = CAPTURE_ARM
            .find("spec.adapters.push(")
            .expect("the capture arm pushes the adapter");
        let gate = CAPTURE_ARM[..push]
            .rfind("if target.variant == TransformerVariant::")
            .expect("the push is gated on the transformer variant");
        assert!(
            CAPTURE_ARM[gate..push].contains("TransformerVariant::Dev"),
            "the capture arm must push DEV_ADAPTER under the Dev arm, not the Distilled one"
        );

        // 3. The enhancer is inserted BEFORE any variant test, i.e. for both variants — which is
        //    why it cancels out of every variant delta and appears in neither row.
        let spec_fn = CAPTURE_ARM
            .find("fn configured_spec(")
            .expect("the capture arm builds the load spec");
        let enhancer = CAPTURE_ARM[spec_fn..]
            .find("\"enhancer\".to_owned()")
            .expect("configured_spec inserts the enhancer component");
        let first_variant_test = CAPTURE_ARM[spec_fn..]
            .find("if target.variant == TransformerVariant::")
            .expect("configured_spec tests the variant");
        assert!(
            enhancer < first_variant_test,
            "the enhancer must be inserted unconditionally; a variant-conditional enhancer would \
             belong in a variant delta row"
        );
        for (_, components) in LTX25_VARIANT_DELTA_COMPONENTS {
            assert!(
                !components.contains(&"enhancer"),
                "the enhancer is resident in both variants and cancels out of the deltas"
            );
        }

        // 4. The production worker agrees: LTX-2.5 Distilled resolves NO adapter.
        let refusal = WORKER
            .find("if ltx25_variant == Some(Ltx25TransformerVariant::Distilled) {")
            .expect("the worker refuses the distill adapter for LTX-2.5 distilled");
        assert!(
            WORKER[refusal..refusal + 200].contains("return Ok(None);"),
            "the worker's distilled arm must return no adapter"
        );
        let distilled_components: BTreeSet<&str> = LTX25_VARIANT_DELTA_COMPONENTS
            .iter()
            .find(|(variant, _)| *variant == "distilled")
            .map(|(_, components)| components.iter().copied().collect())
            .expect("the mirror describes distilled");
        assert!(
            distilled_components.is_empty(),
            "distilled materializes nothing its dev sibling lacks, so its row crosses nothing"
        );
    }

    /// The delta rows are BOUND to the shipped inventory: a zeroed, inflated, re-pointed,
    /// unknown-file or duplicated row fails the load.
    #[test]
    fn a_doctored_component_delta_is_rejected() {
        let raw = PACKAGED_MEMORY_ANCHORS;
        let base: serde_json::Value = serde_json::from_str(raw).expect("store parses");
        assert!(
            base["componentDeltas"]
                .as_array()
                .is_some_and(|rows| !rows.is_empty()),
            "the packaged store must carry component deltas for this harness to interrogate"
        );

        let mut zeroed = base.clone();
        zeroed["componentDeltas"][0]["bytes"] = serde_json::json!(0);
        let error =
            load_memory_anchors(&zeroed.to_string()).expect_err("a zeroed delta must be rejected");
        assert!(
            error.contains("must equal the shipped file sizes"),
            "{error}"
        );

        let mut inflated = base.clone();
        let bytes = inflated["componentDeltas"][0]["bytes"]
            .as_u64()
            .expect("bytes");
        inflated["componentDeltas"][0]["bytes"] = serde_json::json!(bytes + 1);
        let error = load_memory_anchors(&inflated.to_string())
            .expect_err("an inflated delta must be rejected");
        assert!(
            error.contains("must equal the shipped file sizes"),
            "{error}"
        );

        let mut repointed = base.clone();
        repointed["componentDeltas"][0]["source"]["path"] = serde_json::json!("docs/nowhere.json");
        let error = load_memory_anchors(&repointed.to_string())
            .expect_err("a foreign inventory citation must be rejected");
        assert!(
            error.contains("not a compiled weights file inventory"),
            "{error}"
        );

        let mut drifted = base.clone();
        drifted["componentDeltas"][0]["source"]["sha256"] = serde_json::json!("0".repeat(64));
        let error = load_memory_anchors(&drifted.to_string())
            .expect_err("a drifted inventory digest must be rejected");
        assert!(error.contains("inventory digest mismatch"), "{error}");

        let mut unknown = base.clone();
        unknown["componentDeltas"][0]["files"] = serde_json::json!(["not/a/file.safetensors"]);
        let error = load_memory_anchors(&unknown.to_string())
            .expect_err("an unknown file must be rejected");
        assert!(error.contains("does not size"), "{error}");

        // The zero row is representable but not a loophole: dropping a real row's files while
        // keeping its bytes must fail, and so must giving the honest zero row a byte count.
        let mut unfiled = base.clone();
        unfiled["componentDeltas"][0]["files"] = serde_json::json!([]);
        let error = load_memory_anchors(&unfiled.to_string())
            .expect_err("bytes without files must be rejected");
        assert!(error.contains("names no files"), "{error}");

        let zero_index = base["componentDeltas"]
            .as_array()
            .expect("rows")
            .iter()
            .position(|row| {
                row["files"]
                    .as_array()
                    .is_some_and(|files| files.is_empty())
            })
            .expect("the packaged store carries an explicit zero crossing");
        assert_eq!(
            base["componentDeltas"][zero_index]["bytes"].as_u64(),
            Some(0)
        );
        let mut paid_zero = base.clone();
        paid_zero["componentDeltas"][zero_index]["bytes"] = serde_json::json!(1);
        let error = load_memory_anchors(&paid_zero.to_string())
            .expect_err("a priced zero crossing must be rejected");
        assert!(error.contains("names no files"), "{error}");

        let mut duplicated = base.clone();
        let clone = duplicated["componentDeltas"][0].clone();
        duplicated["componentDeltas"]
            .as_array_mut()
            .expect("rows")
            .push(clone);
        let error = load_memory_anchors(&duplicated.to_string())
            .expect_err("a duplicated delta cell must be rejected");
        assert!(error.contains("duplicate component delta"), "{error}");
    }

    /// The new anchor fields are bound, not stored-and-trusted: a doctored phase allocator level,
    /// a dropped decomposition the record reports, and an empty underived reason all fail.
    #[test]
    fn a_doctored_phase_allocator_or_empty_underived_reason_is_rejected() {
        let index = store()
            .anchors
            .iter()
            .position(|anchor| anchor.phase_allocator_envelope_bytes.is_some())
            .expect("an anchor with phase allocator levels");
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("store parses");
        let level = doctored["anchors"][index]["phaseAllocatorEnvelopeBytes"]["denoise"]
            .as_u64()
            .expect("denoise allocator level");
        doctored["anchors"][index]["phaseAllocatorEnvelopeBytes"]["denoise"] =
            serde_json::json!(level + 1);
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("a doctored allocator level must be rejected");
        assert!(
            error.contains("phase allocator envelopes disagree"),
            "{error}"
        );

        let mut dropped: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("store parses");
        dropped["anchors"][index]
            .as_object_mut()
            .expect("anchor object")
            .remove("phaseAllocatorEnvelopeBytes");
        let error = load_memory_anchors(&dropped.to_string())
            .expect_err("dropping a reported decomposition must be rejected");
        assert!(
            error.contains("phase allocator envelopes disagree"),
            "{error}"
        );

        let mut unexplained: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("store parses");
        unexplained["anchors"][index]["underivedReason"] = serde_json::json!("   ");
        let error = load_memory_anchors(&unexplained.to_string())
            .expect_err("an empty underived reason must be rejected");
        assert!(error.contains("without a reason"), "{error}");
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
