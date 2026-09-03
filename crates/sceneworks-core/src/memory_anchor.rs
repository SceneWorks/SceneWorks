//! Measured memory anchors and the analytic peak derivation built on them (sc-22507, epic 22505).
//!
//! One anchor per `(model, quant tier, backend lane)` carries the measured per-phase peak
//! decomposition of a single retained calibration render. Everything else is derived analytically.
//! For the VIDEO lane (LTX-2.5 on MLX): activation terms scale linearly in latent tokens
//! (`area x latent frames`), decode scales in output voxels, attention transients are bounded by
//! the declared rung parameters, and bounded regimes (tiled decode, windowed transformer
//! residency, deferred materialization) substitute architecture-bounded terms for the anchored
//! ones. For the IMAGE lanes (candle and MLX alike) there is exactly one law and no fitted
//! coefficient (sc-22663, epic 22657 E3): per phase, the residue is the anchor's measured peak
//! minus the component bytes resident in that phase under the anchor's regime; the request's
//! regime re-adds its own resident components and scales the residues by architecture facts and
//! geometry — see the *Image derivation law* section and [`MemoryAnchor::derive_phase_peaks`].
//! This replaces grid measurement: a request at a `(geometry, frames)` cell that was never
//! measured is admitted from the derived estimate.
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
// Image derivation law — sc-22663, epic 22657 E3.
//
// ONE law for both image lanes, and no fitted coefficient anywhere in it. The two per-lane image
// laws this replaces (sc-22509's candle per-pixel slopes and the feature-end MLX allocator slopes)
// each fitted three coefficients to ONE model's 768x768 -> 1024x1024 pair, then had to refuse every
// other model ("underived"), clamp every request below 768x768 to that geometry, and refuse any
// composition other than the one the anchor was measured in. Every one of those restrictions was a
// consequence of fitting, and every one is gone here because nothing is fitted:
//
//   residue(phase) = measured phase peak − the component bytes RESIDENT in that phase under the
//                    anchor's own regime
//
// is what one retained render says about the phase's ACTIVATION working set, independent of which
// weights happened to be loaded next to it. A request in any regime then re-adds the components
// ITS regime keeps resident in that phase, and scales the residue by ratios that are architecture
// facts and geometry, never fits:
//
//   * decode under `bounded_decode`: the residue splits into a NON-TILING FLOOR carried unscaled
//     and a per-tile part scaled by (decode chunk pixels / image pixels). The floor is what every
//     bounded decode holds whole-image regardless of the tile: the full-image output accumulator
//     and the full-image blend-weight accumulator of the engine's tile blender (`candle-gen/src/
//     vae_tiling.rs::blend_plan` and the Qwen-Image VAE's `tile_blend_tail` at the pinned
//     revision — one `[B, 3, H, W]` buffer in the activation dtype and one `[1, 1, H, W]` f32
//     buffer), so it is `pixels x (3 x batch x activation width + 4)` and needs the activation
//     width fact; without it the whole residue stays unscaled. The chunk is the LARGER of the two
//     bounding models the pinned engines execute, because the law cannot know per provider which
//     one a request runs: Krea 2 Turbo and Qwen-Image tile the VAE ON DEVICE through
//     `gen_core::tiling::split_spatial` (tiles of `edge²` output pixels placed at stride
//     `edge − overlap`, the halo INSIDE the extent), while Z-Image-Turbo and FLUX transfer
//     `(edge − overlap)` output rows of latents to a whole-frame CPU decode
//     (`bounded_host_latent_transfer`), a band of `(edge − overlap) x width` output pixels. The
//     effective tile fraction is therefore never below `(edge − overlap) x width / pixels`;
//   * denoise: the full score tensor — heads x tokens² x activation width, tokens from the request
//     geometry through the VAE scale and patch size plus the conditioning tokens — is separated out
//     of the residue and replaced by the chunk budget x width under `bounded_attention` (capped at
//     the unchunked tensor: a chunk larger than the tensor cannot cost more than the tensor); the
//     rest of the residue scales linearly in tokens;
//   * transformer resident bytes x window / blocks under `bounded_transformer_residency`;
//   * conditioning: the residue scales with the prompt's conditioning tokens over
//     [`DEFAULT_CONDITIONING_TOKENS`], never below 1.0 (a shorter prompt than the records were
//     measured with is not credited);
//   * geometry through token and pixel counts; batch multiplies both.
//
// A missing fact leaves the residue it would have scaled UNSCALED, so the estimate only ever errs
// large: [`ArchitectureFacts::default()`] prices every rung at the shallow composition. There is no
// lower clamp — a 512x512 request prices below the 1024x1024 anchor, as it should — and no refusal
// by composition: a resident request from a staged anchor re-adds the text encoder through denoise
// and decode and prices ABOVE the staged estimate instead of returning `None`.
//
// DOMAIN. A phase whose measured peak is BELOW the component set it is decomposed against has
// left the law's domain: the counters did not see the weights the anchor's regime says were
// resident, so the subtraction says nothing about the activation working set (clamping it to zero
// would read "the counters missed the weights" as "zero activation" and price a windowed denoise
// at the window's weights alone). Such a request returns `None` and the caller keeps its floor.
// Every packaged MLX image anchor is outside the domain today — its conditioning-phase active
// peak is a fraction of the eager resident set it claims, and so is its conditioning-phase
// ALLOCATOR level (the quantity the MLX fit gate admits on), so decomposing the allocator
// envelope instead would not bring one in (see
// `the_packaged_mlx_anchors_are_outside_the_laws_domain` for both censuses) — so the MLX lane
// entry point prices nothing from the packaged store until an anchor whose counters saw its
// weights is captured.
//
// The measured phase level the law subtracts from is [`MemoryAnchor::phase_active_peak_bytes`],
// the store's byte-bound decomposition. On the candle lane the allocator level equals it in every
// retained record. On MLX the allocator retains cache above the active peak; that envelope is a
// LANE admission term, not an activation residue, and [`MemoryAnchor::derive_mlx_image_phase_peaks`]
// carries it on top of this law rather than folding it in — a proportionally scaled allocator
// level under-brackets the retained 768x768 flux2 denoise levels by 1.5%, exactly the class of
// error a fitted margin used to paper over.
//
// The identity guards the earlier laws carried (backend lane, LTX pipeline axes, a single-frame
// measured geometry) stay, and so does the anchor-vs-request regime guard: a phase the anchor
// measured under a bounded rung carries that rung's bounded working set and cannot price the same
// phase UNBOUNDED, so such a request is refused; the correspondingly-bounded request reuses the
// bounded residue unscaled (the anchor row records no rung parameters, so the ratio is unknowable
// and "unscaled" is the erring-large choice).
// ---------------------------------------------------------------------------------------------

/// Conditioning-token count assumed when a request states none. The Qwen3 / Qwen2.5-VL text
/// encoders the image lanes condition on cap the prompt at 512 tokens (`max_sequence_length` of
/// the reference pipelines), and the retained calibration records do not carry the prompt length
/// they were measured with, so the cap is the documented stand-in on both sides of the ratio.
pub const DEFAULT_CONDITIONING_TOKENS: u32 = 512;

/// Architecture facts of the model an anchor measures, every one optional: a fact that is `None`
/// leaves the residue it would have scaled unscaled (see the section comment). The worker fills
/// these from the live provider contract; a fixture may fill them from the model's config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArchitectureFacts {
    /// Attention heads of the transformer (the score tensor is one `tokens x tokens` matrix per
    /// head).
    pub attention_heads: Option<u32>,
    /// Per-head width. Carried as part of the architecture identity; the score-tensor term is
    /// `heads x tokens²` and does not depend on it.
    pub head_dim: Option<u32>,
    /// Transformer block count — the denominator of the residency-window ratio.
    pub transformer_blocks: Option<u32>,
    /// Latent patch edge (latent pixels per token edge).
    pub patch_size: Option<u32>,
    /// Latent channels. Carried as part of the architecture identity.
    pub latent_channels: Option<u32>,
    /// Output pixels per latent pixel along one edge.
    pub vae_spatial_scale: Option<u32>,
    /// Output frames per latent frame. Still-image anchors measure one frame; carried so a video
    /// anchor can state its temporal scale under the same struct.
    pub vae_temporal_scale: Option<u32>,
    /// Bytes per activation element (2 for bf16/f16, 4 for f32).
    pub activation_dtype_width: Option<u32>,
}

/// The resident byte size of each heavy component of the model an anchor measures, at the anchor's
/// tier — what the law subtracts from a measured phase peak and re-adds per the request's regime.
/// The worker supplies these from the live provider contract's asset facts; an anchor row MAY carry
/// them ([`MemoryAnchor::component_bytes`]) once the extractor can read them off the record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ComponentBytes {
    /// The text/conditioning encoder(s).
    pub conditioning: u64,
    /// The denoising transformer.
    pub transformer: u64,
    /// The latent decoder.
    pub decoder: u64,
}

impl ComponentBytes {
    pub const fn total(self) -> u64 {
        self.conditioning
            .saturating_add(self.transformer)
            .saturating_add(self.decoder)
    }
}

/// The `bounded_decode` tile the request decodes with, in OUTPUT pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeTile {
    pub edge: u32,
    pub overlap: u32,
}

impl DecodeTile {
    /// The output area one ON-DEVICE decode tile materializes: `edge²`. The pinned engine
    /// (`gen_core::tiling::split_spatial` at inference 670dc1f4) places tiles of exactly `edge`
    /// output pixels at a stride of `edge − overlap`, with the overlap halo INSIDE the tile extent,
    /// so the halo is not added on top.
    pub const fn area(self) -> i128 {
        let edge = self.edge as i128;
        edge * edge
    }

    /// The output stride between tiles, `edge − overlap`; `None` when the overlap is not below the
    /// edge (the engines refuse such a tuple, and so does the law).
    pub const fn stride(self) -> Option<u32> {
        if self.overlap < self.edge {
            Some(self.edge - self.overlap)
        } else {
            None
        }
    }

    /// The output-pixel band one HOST-TRANSFER bounded decode holds on the device at a time:
    /// `(edge − overlap)` output rows across the full request width. Z-Image-Turbo and FLUX bound
    /// their decode this way (`bounded_host_latent_transfer` at the pinned revision: the tuple's
    /// difference is the output stride, converted to latent rows and transferred to a whole-frame
    /// CPU decode). `None` when the overlap is not below the edge.
    pub const fn host_transfer_band(self, request_width: u32) -> Option<i128> {
        match self.stride() {
            Some(stride) => Some(stride as i128 * request_width as i128),
            None => None,
        }
    }

    /// The decode chunk the law scales the per-tile decode residue by: the LARGER of the on-device
    /// tile ([`Self::area`]) and the host-transfer band ([`Self::host_transfer_band`]), because the
    /// law cannot know per provider which bounding model the request executes and the larger one
    /// errs large. Callers cap it at the request's own pixel count.
    pub const fn chunk_pixels(self, request_width: u32) -> Option<i128> {
        match self.host_transfer_band(request_width) {
            Some(band) => {
                let area = self.area();
                Some(if band > area { band } else { area })
            }
            None => None,
        }
    }
}

/// The regime the graded request executes in: the residency shape plus the parameters of each
/// engaged bounded rung. `None` on a rung means the rung is not engaged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RequestRegime {
    /// `staged_residency` engaged: the text encoder is dropped before denoise, the transformer
    /// before decode. `false` runs whole-model resident through every phase.
    pub staged: bool,
    /// `bounded_decode` engaged with this tile.
    pub decode_tile: Option<DecodeTile>,
    /// `bounded_attention` engaged with this many concurrently materialized score elements.
    pub attention_chunk_scores: Option<u64>,
    /// `bounded_transformer_residency` engaged with this many resident blocks.
    pub transformer_window: Option<u32>,
}

impl RequestRegime {
    /// The shallow staged composition — `staged_residency` and nothing deeper.
    pub const fn staged() -> Self {
        Self {
            staged: true,
            decode_tile: None,
            attention_chunk_scores: None,
            transformer_window: None,
        }
    }

    /// The whole-model resident composition.
    pub const fn resident() -> Self {
        Self {
            staged: false,
            decode_tile: None,
            attention_chunk_scores: None,
            transformer_window: None,
        }
    }
}

/// One still-image workload for [`MemoryAnchor::derive_phase_peaks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDeriveRequest {
    pub width: u32,
    pub height: u32,
    pub batch: u32,
    /// Conditioning tokens the prompt encodes to; [`DEFAULT_CONDITIONING_TOKENS`] when unknown.
    /// Enters the joint-attention sequence length (so the score tensor and the token-linear
    /// denoise residue see it) and scales the conditioning residue by
    /// `tokens / DEFAULT_CONDITIONING_TOKENS`, never below 1.0: the retained records do not carry
    /// the prompt length they were measured with, so a shorter prompt is not credited.
    pub conditioning_tokens: Option<u32>,
    pub regime: RequestRegime,
}

impl ImageDeriveRequest {
    pub const fn new(width: u32, height: u32, regime: RequestRegime) -> Self {
        Self {
            width,
            height,
            batch: 1,
            conditioning_tokens: None,
            regime,
        }
    }
}

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
    /// `Some` when this anchor VALIDATES its measured point but no FITTED lane law may derive from
    /// it, with the stated reason (epic 22505 feature-end fix round, per-model scoping). The reason
    /// is always a statement about fitting — a single measured geometry supports no per-pixel
    /// coefficient, an axis-free row keys no video variant — so the video law and the lane shims
    /// ([`MemoryAnchor::derive_image_phase_peaks`], [`MemoryAnchor::derive_mlx_image_phase_peaks`])
    /// refuse an anchor carrying it, and the memory matrix publishes the cell as
    /// `Anchored/underived`. The derivation law itself ([`MemoryAnchor::derive_phase_peaks`],
    /// sc-22663) fits nothing and does not consult it. Written by the extractor from the model's
    /// own retained evidence, never a blanket switch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub underived_reason: Option<String>,
    /// The resident byte size of the measured model's components at the anchor's tier
    /// (sc-22663), when the retained record states them. The derivation law takes components as
    /// an explicit argument — the worker supplies the live contract's asset facts — so this field
    /// is a carried fact for readers and for a future extractor that can read a record's contract
    /// snapshot, not a prerequisite: today's retained records state no component bytes and every
    /// packaged row is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component_bytes: Option<ComponentBytes>,
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
    // Component bytes are optional, but a stated set must state something: all-zero components
    // would make every phase's residue the whole measured peak and re-add nothing, i.e. a row that
    // silently prices every regime as the anchor's own.
    if anchor
        .component_bytes
        .is_some_and(|components| components.total() == 0)
    {
        return Err(format!(
            "memory anchor {} states component bytes that sum to zero",
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

/// Per-phase peak estimates. The admission peak is the max over phases. The VIDEO derivation
/// ([`MemoryAnchor::derive_video_phase_peaks`]) widens each phase by
/// [`ANCHOR_ALLOCATOR_ENVELOPE_MARGIN`] before returning it; the IMAGE law
/// ([`MemoryAnchor::derive_phase_peaks`]) widens nothing — it prices measured peaks, component
/// bytes and architecture ratios only — and the worker's `ladder_margin_policy` charges an
/// image-lane anchor derivation the lane's same-cell recapture spread instead.
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

/// The two halves of one IMAGE derivation, `[conditioning, denoise, decode]` each: the components
/// the request's regime holds resident in each phase, and the activation residue the law scales
/// onto them. Their per-phase sum is [`MemoryAnchor::derive_phase_peaks`]; the activation half
/// alone is [`MemoryAnchor::derive_phase_activation_residues`].
#[derive(Debug, Clone, Copy)]
struct ImagePhaseSplit {
    resident: [i128; 3],
    activation: [i128; 3],
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

/// The workload axes of [`MemoryAnchor::derive_image_phase_peaks`], the candle lane's entry point
/// onto the derivation law: geometry and the staging flag only. A composition deeper than staged
/// is priced at the staged working set through this entry point (no rung parameters travel with
/// it); [`ImageDeriveRequest`] carries the full regime for the law itself.
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

    /// The derivation law (sc-22663, epic 22657 E3) — see the *Image derivation law* section
    /// comment for the law itself. This is the ONE law both image lanes price from; the lane
    /// entry points [`Self::derive_image_phase_peaks`] (candle) and
    /// [`Self::derive_mlx_image_phase_peaks`] (MLX) are thin translations onto it.
    ///
    /// `components` are the resident byte sizes of the measured model's text encoder, transformer
    /// and decoder at the anchor's tier (the worker reads them off the live provider contract);
    /// `facts` are the architecture facts the rung ratios need, each optional and each leaving
    /// its residue unscaled when absent.
    ///
    /// IDENTITY GUARD: an anchor carrying LTX pipeline axes or a multi-frame measured geometry is
    /// refused — those are the video law's. Both backend lanes are priced here; the lane shims
    /// pin the lane.
    ///
    /// REGIME GUARD (anchor vs request): a phase the anchor measured under a bounded rung carries
    /// that rung's bounded working set, so it cannot price the same phase UNBOUNDED and such a
    /// request is refused; a request bounded in the same phase reuses the bounded residue
    /// unscaled, the anchor row recording no rung parameters. The anchor's residency shape
    /// (staged or resident) is NOT a guard: it only decides which components its measured peaks
    /// contained, and the request's own shape decides which are re-added.
    ///
    /// DOMAIN GUARD: a phase whose measured peak is below the component set it is decomposed
    /// against is refused (see the section comment) — the residue is never clamped to zero.
    ///
    /// Returns `None` on degenerate geometry, a zero rung parameter or a tile whose overlap is not
    /// below its edge, a refused identity/regime, a phase outside the law's domain, or a phase
    /// that prices to zero bytes (a zero peak would admit anything).
    pub fn derive_phase_peaks(
        &self,
        request: &ImageDeriveRequest,
        components: ComponentBytes,
        facts: ArchitectureFacts,
    ) -> Option<AnchorDerivedPhases> {
        let split = self.derive_image_phase_split(request, components, facts)?;
        Some(AnchorDerivedPhases {
            conditioning: positive(split.resident[0] + split.activation[0])?,
            denoise: positive(split.resident[1] + split.activation[1])?,
            decode: positive(split.resident[2] + split.activation[2])?,
        })
    }

    /// The ACTIVATION half of [`Self::derive_phase_peaks`], phase by phase: the anchor's measured
    /// residue after the components are subtracted, re-scaled for the request's geometry, batch,
    /// prompt length and engaged rungs — and NOT re-adding any component byte.
    ///
    /// This is what a caller that already prices the model's wired weights itself needs (sc-22665,
    /// epic 22657 E4): the MLX worker's estimate floor composes its own contract-decomposed weights
    /// term with this residue instead of the lane's geometry-blind generic headroom, so the
    /// request's tile, attention chunk and transformer window reach the estimate. A caller that
    /// wants the whole priced peak, components included, calls [`Self::derive_phase_peaks`].
    ///
    /// Refuses exactly what [`Self::derive_phase_peaks`] refuses, the per-phase positivity of the
    /// FULL estimate included: a residue the law would not have stood behind as part of a whole
    /// peak is not handed out on its own.
    pub fn derive_phase_activation_residues(
        &self,
        request: &ImageDeriveRequest,
        components: ComponentBytes,
        facts: ArchitectureFacts,
    ) -> Option<AnchorPhaseBytes> {
        let split = self.derive_image_phase_split(request, components, facts)?;
        for phase in 0..3 {
            positive(split.resident[phase] + split.activation[phase])?;
        }
        Some(AnchorPhaseBytes {
            conditioning: u64::try_from(split.activation[0]).ok()?,
            denoise: u64::try_from(split.activation[1]).ok()?,
            decode: u64::try_from(split.activation[2]).ok()?,
        })
    }

    /// The law itself, kept as the two halves its consumers need separately: the components the
    /// REQUEST's regime holds resident in each phase, and the activation residue the law scales
    /// onto them. `[conditioning, denoise, decode]` in both, in `i128` because the sum is what the
    /// positivity guard is defined on — an intermediate is never clamped.
    fn derive_image_phase_split(
        &self,
        request: &ImageDeriveRequest,
        components: ComponentBytes,
        facts: ArchitectureFacts,
    ) -> Option<ImagePhaseSplit> {
        if request.width == 0 || request.height == 0 || request.batch == 0 {
            return None;
        }
        if self.transformer_variant.is_some() || self.decoder.is_some() || self.geometry.frames != 1
        {
            return None;
        }
        let regime = request.regime;
        let measured = self.measured_regime;
        if (measured.decode_tiled && regime.decode_tile.is_none())
            || (measured.attention_chunked && regime.attention_chunk_scores.is_none())
            || (measured.transformer_windowed && regime.transformer_window.is_none())
        {
            return None;
        }
        if regime
            .decode_tile
            .is_some_and(|tile| tile.edge == 0 || tile.stride().is_none())
            || regime.attention_chunk_scores == Some(0)
            || regime.transformer_window == Some(0)
        {
            return None;
        }

        let conditioning = i128::from(components.conditioning);
        let transformer = i128::from(components.transformer);
        let decoder = i128::from(components.decoder);
        let everything = conditioning + transformer + decoder;

        // Residue per phase: the measured peak minus the components resident in that phase under
        // the ANCHOR's regime. Staged drops the text encoder before denoise and the transformer
        // before decode (the candle three-stage path); resident holds all three throughout. A
        // measured peak BELOW its resident set is outside the law's domain (the counters did not
        // see the weights the regime claims resident, so the subtraction says nothing about the
        // activation working set) and refuses the request rather than clamping to zero.
        let (anchor_cond, anchor_den, anchor_dec) = if measured.staged {
            (conditioning, transformer, decoder)
        } else {
            (everything, everything, everything)
        };
        let residue = |peak: u64, resident: i128| {
            let residue = i128::from(peak) - resident;
            (residue >= 0).then_some(residue)
        };
        let peaks = self.phase_active_peak_bytes;
        let cond_residue = residue(peaks.conditioning, anchor_cond)?;
        let den_residue = residue(peaks.denoise, anchor_den)?;
        let dec_residue = residue(peaks.decode, anchor_dec)?;

        // Geometry. Pixel counts need no fact; token counts need the VAE scale and patch size.
        let batch = i128::from(request.batch);
        let anchor_pixels = i128::from(self.geometry.width) * i128::from(self.geometry.height);
        let request_pixels = i128::from(request.width) * i128::from(request.height);
        // The anchor's records do not carry the prompt length they were measured with, so the
        // anchor side of every token ratio is the documented default; the request side is its
        // own count floored at that default (a shorter prompt is never credited — erring large).
        let default_tokens = i128::from(DEFAULT_CONDITIONING_TOKENS);
        let conditioning_tokens = i128::from(
            request
                .conditioning_tokens
                .unwrap_or(DEFAULT_CONDITIONING_TOKENS),
        )
        .max(default_tokens);
        let token_stride = match (facts.vae_spatial_scale, facts.patch_size) {
            (Some(scale), Some(patch)) if scale > 0 && patch > 0 => {
                Some(i128::from(scale) * i128::from(patch))
            }
            _ => None,
        };
        // Joint-attention sequence length of one sample: image tokens plus conditioning tokens.
        let tokens = |width: u32, height: u32, conditioning_tokens: i128| {
            token_stride.map(|stride| {
                div_ceil_i128(i128::from(width), stride) * div_ceil_i128(i128::from(height), stride)
                    + conditioning_tokens
            })
        };
        let anchor_tokens = tokens(self.geometry.width, self.geometry.height, default_tokens);
        let request_tokens = tokens(request.width, request.height, conditioning_tokens);
        // Linear activation ratio: tokens when the facts allow, else pixels; batch multiplies.
        let (linear_num, linear_den) = match (request_tokens, anchor_tokens) {
            (Some(request_tokens), Some(anchor_tokens)) if anchor_tokens > 0 => {
                (request_tokens * batch, anchor_tokens)
            }
            _ => (request_pixels * batch, anchor_pixels),
        };

        // Resident components under the REQUEST's regime. The window applies wherever the
        // transformer is counted: windowed residency materializes one window at a time for the
        // whole render, not just inside denoise.
        let windowed_transformer = match (regime.transformer_window, facts.transformer_blocks) {
            (Some(window), Some(blocks)) if blocks > 0 => scale_up(
                transformer,
                i128::from(window.min(blocks)),
                i128::from(blocks),
            ),
            _ => transformer,
        };
        let (request_cond, request_den, request_dec) = if regime.staged {
            (conditioning, windowed_transformer, decoder)
        } else {
            let all = conditioning + windowed_transformer + decoder;
            (all, all, all)
        };

        // Conditioning: prompt-shaped, so the residue scales with batch and with the prompt's
        // conditioning tokens over the default the records were measured against — never below
        // 1.0, since the records do not state their prompt length. The retained resident/staged
        // candle pairs measure it flat across 768x768 -> 1024x1024.
        let cond_activation = scale_up(cond_residue * batch, conditioning_tokens, default_tokens);

        // Denoise. With the score facts the residue splits into the full score tensor at the
        // anchor geometry (capped at the residue — the anchor cannot have held more than it
        // measured) and the rest; the rest scales linearly in tokens, the score tensor is
        // re-priced at the request geometry, or replaced by the chunk budget when chunked.
        // Without the facts the whole residue is treated as the worst-scaling term — quadratic
        // in tokens for growth, linear for shrink — and a requested chunk leaves it unscaled.
        let score_bytes = |width: u32, height: u32, conditioning: i128, batch: i128| {
            let heads = i128::from(facts.attention_heads?);
            let element = i128::from(facts.activation_dtype_width?);
            let tokens = tokens(width, height, conditioning)?;
            Some(heads * tokens * tokens * element * batch)
        };
        let den_activation =
            match score_bytes(self.geometry.width, self.geometry.height, default_tokens, 1) {
                Some(_) if measured.attention_chunked => {
                    // The anchor's own denoise was chunked: its residue already holds a chunk
                    // workspace instead of a score tensor, so there is nothing to separate out.
                    scale_up(den_residue, linear_num, linear_den)
                }
                Some(anchor_scores) => {
                    let non_score = den_residue - anchor_scores.min(den_residue);
                    // The chunk budget is capped at the unchunked tensor at the request geometry: a
                    // chunk larger than the whole tensor materializes the tensor, never more, so the
                    // chunked rung can never price above the unchunked one.
                    let unchunked_scores =
                        score_bytes(request.width, request.height, conditioning_tokens, batch)?;
                    let score_term = match regime.attention_chunk_scores {
                        Some(chunk) => (i128::from(chunk)
                            * i128::from(facts.activation_dtype_width.unwrap_or(1)))
                        .min(unchunked_scores),
                        None => unchunked_scores,
                    };
                    scale_up(non_score, linear_num, linear_den) + score_term
                }
                None => {
                    let quadratic_num = request_pixels * request_pixels * batch;
                    let quadratic_den = anchor_pixels * anchor_pixels;
                    let linear = scale_up(den_residue, request_pixels * batch, anchor_pixels);
                    let quadratic = scale_up(den_residue, quadratic_num, quadratic_den);
                    linear.max(quadratic)
                }
            };

        // Decode: pixel-shaped, so the residue scales in output pixels and batch. Under
        // `bounded_decode` (from an anchor measured untiled — an anchor measured tiled already
        // holds the tile working set and is not re-tiled) the scaled residue splits into the
        // NON-TILING FLOOR, carried unscaled, and the per-tile remainder, scaled by the decode
        // chunk's fraction of the image. The floor is the tile blender's two full-image
        // accumulators — output `[B, 3, H, W]` in the activation dtype plus blend weights
        // `[1, 1, H, W]` in f32 (`vae_tiling.rs::blend_plan` at the pin) — so it needs the
        // activation width; without that fact the floor is unknowable and the whole residue stays
        // unscaled. The chunk is the larger of the on-device tile and the host-transfer band
        // (`DecodeTile::chunk_pixels`), capped at the whole image. See the section comment.
        let mut dec_scaled = scale_up(dec_residue, request_pixels * batch, anchor_pixels);
        if let (Some(tile), false) = (regime.decode_tile, measured.decode_tiled) {
            if let Some(element) = facts.activation_dtype_width.filter(|width| *width > 0) {
                let floor =
                    (request_pixels * (3 * batch * i128::from(element) + 4)).min(dec_scaled);
                let chunk = tile.chunk_pixels(request.width)?.min(request_pixels);
                dec_scaled = floor + scale_up(dec_scaled - floor, chunk, request_pixels);
            }
        }
        Some(ImagePhaseSplit {
            resident: [request_cond, request_den, request_dec],
            activation: [cond_activation, den_activation, dec_scaled],
        })
    }

    /// The candle image lane's entry point onto [`Self::derive_phase_peaks`] — sc-22509's
    /// signature, which the worker's candle admission still calls until the lane's own story
    /// rewires it with the request's full regime and the contract's architecture facts.
    ///
    /// Pins the lane (a non-candle anchor is refused) and keeps the shallow-anchor asymmetry the
    /// candle consumers were written against: the anchor must be the staged composition with no
    /// deeper rung engaged, and the graded candidate must engage `staged_residency`. With no rung
    /// parameters and no facts the law prices every composition containing the anchor's at the
    /// shallow staged working set, which upper-bounds the deeper rungs. An anchor carrying
    /// [`MemoryAnchor::underived_reason`] is refused, as before.
    pub fn derive_image_phase_peaks(
        &self,
        request: AnchorImageDeriveRequest,
        components: ComponentBytes,
    ) -> Option<AnchorDerivedPhases> {
        if self.underived_reason.is_some() || self.backend != AnchorBackend::Candle {
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
        self.derive_phase_peaks(
            &ImageDeriveRequest::new(request.width, request.height, RequestRegime::staged()),
            components,
            ArchitectureFacts::default(),
        )
    }

    /// The MLX image lane's entry point onto [`Self::derive_phase_peaks`] — the feature-end
    /// signature the worker's MLX ladder still calls until the lane's own story rewires it.
    ///
    /// Pins the lane and keeps its guards: the anchor must be the eager, unbounded, resident
    /// composition (the widest the lane executes, so the resident derivation upper-bounds every
    /// optimized composition of the cell), must report a per-phase allocator decomposition, and
    /// must not carry [`MemoryAnchor::underived_reason`].
    ///
    /// Returns per-phase ALLOCATOR levels, the quantity MLX admission covers: the law's active
    /// estimate plus the anchor's measured allocator envelope above its active peak, phase by
    /// phase. The envelope is retained cache, which the allocator keeps at smaller geometries and
    /// grows at larger ones, so it is carried unscaled for a request at or below the anchor
    /// geometry and scaled by the pixel ratio above it. It is a lane term, not a residue of the
    /// law — see the *Image derivation law* section comment.
    ///
    /// DOMAIN, at this pin: every packaged MLX image anchor reports a conditioning-phase active
    /// peak far below the eager resident set its tier's component bytes state (the MLX active
    /// counters did not see the weights), so the law refuses each of them and this entry point
    /// returns `None` for the packaged store — the lane keeps its floor until an anchor whose
    /// counters saw its weights is captured (`the_packaged_mlx_anchors_are_outside_the_laws_domain`).
    pub fn derive_mlx_image_phase_peaks(
        &self,
        request: AnchorMlxImageDeriveRequest,
        components: ComponentBytes,
    ) -> Option<AnchorDerivedPhases> {
        if self.underived_reason.is_some() || self.backend != AnchorBackend::Mlx {
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
        let active = self.derive_phase_peaks(
            &ImageDeriveRequest::new(request.width, request.height, RequestRegime::resident()),
            components,
            ArchitectureFacts::default(),
        )?;
        let anchor_pixels = i128::from(self.geometry.width) * i128::from(self.geometry.height);
        let request_pixels = i128::from(request.width) * i128::from(request.height);
        let envelope = |allocator: u64, active_peak: u64| {
            let envelope = i128::from(allocator.saturating_sub(active_peak));
            if request_pixels > anchor_pixels {
                scale_up(envelope, request_pixels, anchor_pixels)
            } else {
                envelope
            }
        };
        let peaks = self.phase_active_peak_bytes;
        Some(AnchorDerivedPhases {
            conditioning: positive(
                i128::from(active.conditioning)
                    + envelope(allocators.conditioning, peaks.conditioning),
            )?,
            denoise: positive(
                i128::from(active.denoise) + envelope(allocators.denoise, peaks.denoise),
            )?,
            decode: positive(
                i128::from(active.decode) + envelope(allocators.decode, peaks.decode),
            )?,
        })
    }
}

/// `bytes x num / den`, rounded UP so a scaled residue never loses a byte to integer division.
fn scale_up(bytes: i128, num: i128, den: i128) -> i128 {
    if den <= 0 || bytes <= 0 || num <= 0 {
        return 0;
    }
    div_ceil_i128(bytes * num, den)
}

/// A derived phase must be a positive byte count; zero or negative is refused, not clamped.
fn positive(bytes: i128) -> Option<u64> {
    (bytes > 0).then(|| u64::try_from(bytes).ok()).flatten()
}

/// The workload axes of [`MemoryAnchor::derive_mlx_image_phase_peaks`]: geometry only. The lane
/// entry point prices the resident composition, which upper-bounds every composition the lane can
/// execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorMlxImageDeriveRequest {
    pub width: u32,
    pub height: u32,
}

/// Ceiling division on non-negative `i128` operands (`i128::div_ceil` is not stable).
fn div_ceil_i128(numerator: i128, denominator: i128) -> i128 {
    (numerator + denominator - 1) / denominator
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
    // Image derivation law (sc-22663, epic 22657 E3) — fixtures.
    // -------------------------------------------------------------------------------------

    const KREA_CANDLE_CORPUS_PATH: &str = "docs/generated/krea-candle-five-rung-sc-11045.json";

    /// The sc-15859 Z-Image-Turbo q4 candle anchor capture: `staged_residency` at 1024x1024,
    /// conditioning 3.10 GB / denoise 8.05 GB / decode 11.74 GB device peak deltas. Not a
    /// compiled-in store source (the store is regenerated at epic end), so the test reads the
    /// retained file at run time — deliberately not `include_str!`, which would make a test-only
    /// fixture a production embed every Docker builder context must copy — and builds the anchor
    /// row from the record.
    const Z_IMAGE_Q4_CANDLE_ANCHOR_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/calibration/sc-15859/z-image-turbo-q4-candle-anchor.json"
    );

    fn z_image_q4_candle_anchor_raw() -> String {
        std::fs::read_to_string(Z_IMAGE_Q4_CANDLE_ANCHOR_PATH)
            .unwrap_or_else(|error| panic!("{Z_IMAGE_Q4_CANDLE_ANCHOR_PATH}: {error}"))
    }

    /// Z-Image-Turbo q4 component bytes on the candle lane: the `SceneWorks/z-image-turbo-mlx`
    /// q4 tier the retained record's `artifact` names (revision bb2bc989), whose subdirectories
    /// measure text_encoder (Qwen3) 2.26 GB, transformer 3.47 GB, vae 0.16 GB on disk. The
    /// record carries no per-component inventory, so the figures are bound to the tier's
    /// packaged `estimatedSizeBytes` instead:
    /// `the_fixture_component_bytes_sum_to_their_packaged_tier_sizes` recomputes the sum against
    /// the manifest within 2%.
    const Z_IMAGE_Q4_COMPONENTS: ComponentBytes = ComponentBytes {
        conditioning: 2_260_000_000,
        transformer: 3_470_000_000,
        decoder: 160_000_000,
    };

    /// Z-Image architecture facts (`candle_transformers::models::z_image::transformer::Config::
    /// z_image_turbo`): 30 heads of 128, 30 blocks, patch 2, 16 latent channels, x8 VAE, bf16
    /// activations.
    const Z_IMAGE_FACTS: ArchitectureFacts = ArchitectureFacts {
        attention_heads: Some(30),
        head_dim: Some(128),
        transformer_blocks: Some(30),
        patch_size: Some(2),
        latent_channels: Some(16),
        vae_spatial_scale: Some(8),
        vae_temporal_scale: Some(1),
        activation_dtype_width: Some(2),
    };

    /// Krea 2 Turbo architecture facts (`candle-gen-krea/src/config.rs` at the pinned inference
    /// revision): 48 heads of 128, 28 single-stream blocks, patch 2 over the 16-channel x8
    /// Qwen-Image VAE, bf16 activations.
    const KREA_FACTS: ArchitectureFacts = ArchitectureFacts {
        attention_heads: Some(48),
        head_dim: Some(128),
        transformer_blocks: Some(28),
        patch_size: Some(2),
        latent_channels: Some(16),
        vae_spatial_scale: Some(8),
        vae_temporal_scale: Some(1),
        activation_dtype_width: Some(2),
    };

    /// Krea 2 Turbo q4 component bytes on the candle lane, read off the retained five-rung
    /// corpus itself: the resident-minus-staged DENOISE delta is the text encoder that staging
    /// drops (2,818,572,288 bytes; the decode delta agrees byte for byte), the resident-minus-
    /// staged CONDITIONING delta is the DiT plus VAE that staging has not yet loaded
    /// (8,891,924,480), split at the Qwen-Image VAE's bf16 safetensors size (~254 MB).
    const KREA_Q4_COMPONENTS: ComponentBytes = ComponentBytes {
        conditioning: 2_818_572_288,
        transformer: 8_638_117_888,
        decoder: 253_806_592,
    };

    /// The Z-Image / Krea candle fully-engaged rung parameters the providers publish
    /// (`bounded_decode` 512/128, `bounded_attention` 64 Mi scores for Z-Image, one-block
    /// transformer window).
    const FULLY_ENGAGED: RequestRegime = RequestRegime {
        staged: true,
        decode_tile: Some(DecodeTile {
            edge: 512,
            overlap: 128,
        }),
        attention_chunk_scores: Some(64 * 1024 * 1024),
        transformer_window: Some(1),
    };

    fn at(width: u32, height: u32, regime: RequestRegime) -> ImageDeriveRequest {
        ImageDeriveRequest::new(width, height, regime)
    }

    /// Build the anchor row for the single record of a memory-v5 candle capture file, exactly as
    /// the extractor would (identity, geometry, regime, the three device peak deltas).
    fn candle_record_anchor(raw: &str) -> MemoryAnchor {
        let source: serde_json::Value = serde_json::from_str(raw).expect("capture parses");
        let record = &source["records"][0];
        let target = &record["target"];
        let measured: BTreeMap<&str, u64> = record["diagnostics"]["measurements"]
            .as_array()
            .expect("measurements")
            .iter()
            .filter_map(|entry| Some((entry["name"].as_str()?, entry["value"].as_u64()?)))
            .collect();
        let engaged: Vec<&str> = record["strategy"]["engagedRungs"]
            .as_array()
            .expect("engaged rungs")
            .iter()
            .filter_map(|rung| rung.as_str())
            .collect();
        let str_at = |value: &serde_json::Value, key: &str| {
            value[key]
                .as_str()
                .unwrap_or_else(|| panic!("{key}"))
                .to_owned()
        };
        MemoryAnchor {
            id: format!("{}:{}", str_at(target, "modelId"), str_at(record, "id")),
            model_id: str_at(target, "modelId"),
            model_family: String::new(),
            route: str_at(target, "provider"),
            provider: str_at(target, "provider"),
            backend: match record["backend"].as_str() {
                Some("candle") => AnchorBackend::Candle,
                _ => AnchorBackend::Mlx,
            },
            tier: str_at(target, "tier"),
            transformer_variant: None,
            decoder: None,
            mode: str_at(target, "mode"),
            overlay: None,
            reference_count: 0,
            load_shape: match record["loadShape"].as_str() {
                Some("eager_materialization") => AnchorLoadShape::EagerMaterialization,
                _ => AnchorLoadShape::DeferredMaterialization,
            },
            measured_regime: AnchorMeasuredRegime {
                decode_tiled: engaged.contains(&"bounded_decode"),
                transformer_windowed: engaged.contains(&"bounded_transformer_residency"),
                staged: engaged.contains(&"staged_residency"),
                attention_chunked: engaged.contains(&"bounded_attention"),
            },
            source: AnchorSource {
                path: String::new(),
                sha256: String::new(),
                record_id: str_at(record, "id"),
                calibration_fingerprint: str_at(record, "calibrationFingerprint"),
                loader_closure_digest: "0".repeat(64),
            },
            geometry: AnchorGeometry {
                width: target["geometry"]["width"].as_u64().expect("width") as u32,
                height: target["geometry"]["height"].as_u64().expect("height") as u32,
                frames: target["geometry"]["frames"].as_u64().expect("frames") as u32,
                fps: None,
            },
            phase_active_peak_bytes: AnchorPhaseBytes {
                conditioning: measured["conditioningDevicePeakDelta"],
                denoise: measured["denoiseDevicePeakDelta"],
                decode: measured["decodeDevicePeakDelta"],
            },
            phase_allocator_envelope_bytes: None,
            overall_allocator_envelope_bytes: measured["overallDevicePeakDelta"],
            underived_reason: None,
            component_bytes: None,
        }
    }

    fn z_image_q4_anchor() -> MemoryAnchor {
        let anchor = candle_record_anchor(&z_image_q4_candle_anchor_raw());
        // Shape of the fixture the story cites: the shallow staged composition at 1024x1024.
        assert_eq!(anchor.model_id, "z_image_turbo");
        assert_eq!(anchor.tier, "q4");
        assert_eq!(anchor.backend, AnchorBackend::Candle);
        assert_eq!(
            (anchor.geometry.width, anchor.geometry.height),
            (1024, 1024)
        );
        assert_eq!(
            anchor.measured_regime,
            AnchorMeasuredRegime {
                decode_tiled: false,
                transformer_windowed: false,
                staged: true,
                attention_chunked: false,
            }
        );
        assert_eq!(anchor.phase_active_peak_bytes.conditioning, 3_097_493_504);
        assert_eq!(anchor.phase_active_peak_bytes.denoise, 8_050_966_528);
        assert_eq!(anchor.phase_active_peak_bytes.decode, 11_741_954_048);
        anchor
    }

    fn z_image_at(request: &ImageDeriveRequest) -> AnchorDerivedPhases {
        z_image_q4_anchor()
            .derive_phase_peaks(request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
            .unwrap_or_else(|| panic!("{request:?} must be derivable"))
    }

    /// One retained candle capture: the rung composition it executed, its parameters, and its
    /// three measured phase peaks.
    struct CandleCorpusRecord {
        id: String,
        engaged: Vec<String>,
        regime: RequestRegime,
        width: u32,
        height: u32,
        conditioning: u64,
        denoise: u64,
        decode: u64,
    }

    impl CandleCorpusRecord {
        fn peak(&self) -> u64 {
            self.conditioning.max(self.denoise).max(self.decode)
        }
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
                let engaged: Vec<String> = record["strategy"]["engagedRungs"]
                    .as_array()
                    .expect("engaged rungs")
                    .iter()
                    .filter_map(|rung| rung.as_str().map(str::to_owned))
                    .collect();
                let parameters = &record["strategy"]["parameters"];
                let has = |rung: &str| engaged.iter().any(|engaged| engaged == rung);
                let regime = RequestRegime {
                    staged: has("staged_residency"),
                    decode_tile: has("bounded_decode").then(|| DecodeTile {
                        edge: parameters["decodeTileEdge"].as_u64().expect("tile edge") as u32,
                        overlap: parameters["decodeOverlap"].as_u64().expect("overlap") as u32,
                    }),
                    attention_chunk_scores: has("bounded_attention")
                        .then(|| parameters["attentionChunkSize"].as_u64().expect("chunk")),
                    transformer_window: has("bounded_transformer_residency").then(|| {
                        parameters["transformerWindowSize"]
                            .as_u64()
                            .expect("window") as u32
                    }),
                };
                CandleCorpusRecord {
                    id: record["id"].as_str().expect("record id").to_owned(),
                    engaged,
                    regime,
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

    // -------------------------------------------------------------------------------------
    // Image derivation law — the story's acceptance criteria.
    // -------------------------------------------------------------------------------------

    /// AC 1: from the sc-15859 q4 staged anchor and the Z-Image facts, the fully engaged
    /// composition (staged + bounded_decode 512/128 + bounded_attention 64 Mi + window 1) at
    /// 1024x1024 prices in [4, 6] GB overall; each deeper rung's per-phase peaks are at or below
    /// the shallower rung's; and the resident composition prices ABOVE staged instead of `None`.
    #[test]
    fn z_image_q4_rungs_price_from_the_staged_anchor_in_the_restated_window_and_in_order() {
        const GB: u64 = 1_000_000_000;
        let staged = RequestRegime::staged();
        let tiled = RequestRegime {
            decode_tile: FULLY_ENGAGED.decode_tile,
            ..staged
        };
        let chunked = RequestRegime {
            attention_chunk_scores: FULLY_ENGAGED.attention_chunk_scores,
            ..tiled
        };
        let windowed = RequestRegime {
            transformer_window: FULLY_ENGAGED.transformer_window,
            ..chunked
        };
        assert_eq!(windowed, FULLY_ENGAGED);
        let ladder =
            [staged, tiled, chunked, windowed].map(|regime| z_image_at(&at(1024, 1024, regime)));

        // The staged derivation at the anchor's own geometry and composition IS the anchor.
        let anchor = z_image_q4_anchor();
        assert_eq!(
            ladder[0].conditioning,
            anchor.phase_active_peak_bytes.conditioning
        );
        assert_eq!(ladder[0].denoise, anchor.phase_active_peak_bytes.denoise);
        assert_eq!(ladder[0].decode, anchor.phase_active_peak_bytes.decode);

        // Each deeper rung is at or below the shallower one, phase for phase — and the rung that
        // engages a phase's bound moves that phase strictly, so the ratios are known to bite.
        for pair in ladder.windows(2) {
            let (shallow, deep) = (pair[0], pair[1]);
            assert!(deep.conditioning <= shallow.conditioning);
            assert!(deep.denoise <= shallow.denoise);
            assert!(deep.decode <= shallow.decode);
        }
        assert!(
            ladder[1].decode < ladder[0].decode,
            "the tile ratio must bite"
        );
        assert!(ladder[2].denoise < ladder[1].denoise, "the chunk must bite");
        assert!(
            ladder[3].denoise < ladder[2].denoise,
            "the window must bite"
        );

        let fully_engaged = ladder[3];
        let overall = fully_engaged.peak_bytes();
        // The restated AC-1 window (review of sc-22663): [3, 6] GB overall, strictly below the
        // staged anchor's overall, and never below the components the engaged composition keeps
        // resident in any phase — the text encoder plus its conditioning residue, the windowed
        // DiT share, the VAE. No pad sits anywhere in this arithmetic.
        assert!(
            (3 * GB..=6 * GB).contains(&overall),
            "the fully engaged 1024x1024 composition must price in [3, 6] GB overall, got \
             {overall} ({fully_engaged:?})"
        );
        assert!(overall < ladder[0].peak_bytes());
        let components = Z_IMAGE_Q4_COMPONENTS;
        let cond_residue = anchor.phase_active_peak_bytes.conditioning - components.conditioning;
        assert!(overall >= components.conditioning + cond_residue);
        assert!(overall >= components.transformer.div_ceil(30) + components.decoder);
        assert!(fully_engaged.denoise >= components.transformer.div_ceil(30));
        assert!(fully_engaged.decode >= components.decoder);
        // The arithmetic behind that window, stated so a drift is legible: one resident block
        // (3.47 GB / 30), the non-score denoise residue (8.05 - 3.47 GB minus the 30 x 4608² x
        // 2 B score tensor) plus the 64 Mi x 2 B chunk; and, over the VAE, the decode residue
        // (11.74 - 0.16 GB) split into the unscaled blender floor (1024² x (3 x 2 B + 4 B)) and
        // the remainder scaled by the decode chunk — the larger of the 512² on-device tile and
        // the (512 - 128) x 1024 host-transfer band, i.e. 384 x 1024 = 3/8 of the image.
        assert_eq!(fully_engaged.conditioning, 3_097_493_504);
        assert_eq!(
            fully_engaged.denoise,
            115_666_667 + 3_306_946_688 + 134_217_728
        );
        let blender_floor = 1_048_576 * (3 * 2 + 4);
        assert_eq!(
            fully_engaged.decode,
            160_000_000
                + blender_floor
                + div_ceil_i128(
                    (11_581_954_048 - blender_floor as i128) * 393_216,
                    1_048_576
                ) as u64
        );
        assert_eq!(fully_engaged.decode, 4_509_786_368);
        assert_eq!(overall, fully_engaged.decode);

        // Resident from the staged anchor: priced, and strictly above staged in every phase.
        let resident = z_image_at(&at(1024, 1024, RequestRegime::resident()));
        assert!(resident.conditioning > ladder[0].conditioning);
        assert!(resident.denoise > ladder[0].denoise);
        assert!(resident.decode > ladder[0].decode);
        // …by exactly the components staging keeps out of each phase.
        let components = Z_IMAGE_Q4_COMPONENTS;
        assert_eq!(
            resident.conditioning - ladder[0].conditioning,
            components.transformer + components.decoder
        );
        assert_eq!(
            resident.denoise - ladder[0].denoise,
            components.conditioning + components.decoder
        );
        assert_eq!(
            resident.decode - ladder[0].decode,
            components.conditioning + components.transformer
        );
    }

    /// AC 2, candle bullet: a 512x512 request prices below the 1024x1024 anchor — there is no
    /// lower clamp. (The MLX bullet is `the_packaged_mlx_anchors_are_outside_the_laws_domain`
    /// and `a_resident_mlx_anchor_whose_counters_saw_its_weights_prices_the_ladder_in_order`.)
    #[test]
    fn a_smaller_request_prices_below_the_anchor_with_no_lower_clamp() {
        let staged = RequestRegime::staged();
        let small = z_image_at(&at(512, 512, staged));
        let mid = z_image_at(&at(768, 768, staged));
        let anchored = z_image_at(&at(1024, 1024, staged));
        let large = z_image_at(&at(1536, 1536, staged));
        assert!(small.peak_bytes() < anchored.peak_bytes());
        assert!(small.denoise < anchored.denoise && small.decode < anchored.decode);
        // No clamp: 512x512 sits strictly below 768x768, which sits strictly below the anchor.
        assert!(small.denoise < mid.denoise && small.decode < mid.decode);
        assert!(mid.denoise < anchored.denoise && mid.decode < anchored.decode);
        assert!(anchored.peak_bytes() < large.peak_bytes());
        // …and never below the components the composition holds resident.
        assert!(small.conditioning >= Z_IMAGE_Q4_COMPONENTS.conditioning);
        assert!(small.denoise >= Z_IMAGE_Q4_COMPONENTS.transformer);
        assert!(small.decode >= Z_IMAGE_Q4_COMPONENTS.decoder);
        // Area, not aspect.
        assert_eq!(
            z_image_at(&at(1344, 768, staged)),
            z_image_at(&at(768, 1344, staged))
        );
    }

    /// The packaged tier size (`estimatedSizeBytes` of the `variant == tier` download row) of one
    /// builtin model — the resident set an eager, whole-model anchor of that tier claims.
    fn packaged_tier_bytes(model_id: &str, tier: &str) -> u64 {
        let raw = crate::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, contents)| crate::jsonc::strip_jsonc_comments(contents))
            .expect("the builtin models manifest is compiled in");
        let manifest: serde_json::Value = serde_json::from_str(&raw).expect("manifest parses");
        manifest["models"]
            .as_array()
            .expect("models")
            .iter()
            .find(|model| model["id"].as_str() == Some(model_id))
            .unwrap_or_else(|| panic!("{model_id} is a builtin model"))["downloads"]
            .as_array()
            .unwrap_or_else(|| panic!("{model_id} declares downloads"))
            .iter()
            .find(|download| download["variant"].as_str() == Some(tier))
            .and_then(|download| download["estimatedSizeBytes"].as_u64())
            .unwrap_or_else(|| panic!("{model_id} {tier} declares estimatedSizeBytes"))
    }

    /// AC 2, MLX bullet — what the packaged store can and cannot support. The story asked for
    /// the packaged `qwen_image` MLX resident anchor to price rung 4 below rung 2; under the
    /// domain guard (review of sc-22663: a residue is never clamped to zero) NO packaged MLX
    /// image anchor is derivable with its tier's real component bytes, because every one of them
    /// reports a conditioning-phase active peak that is a fraction of the eager resident set it
    /// claims — the MLX active counters did not see the weights. The numbers, so the refusal is
    /// legible: conditioning active peaks of 1.26 MB (flux2_dev q4 and q8), 43.9 KB (qwen_image
    /// bf16/q4/q8) and 2.27 GB (z_image_turbo q4) against packaged resident sets of 33.6 GB,
    /// 55.0 GB, 57.7 / 28.4 / 38.6 GB and 5.9 GB. The lane entry point therefore prices nothing
    /// from the packaged store; the rung ordering itself is asserted on an in-domain anchor in
    /// `a_resident_mlx_anchor_whose_counters_saw_its_weights_prices_the_ladder_in_order`.
    ///
    /// ALLOCATOR BASIS (second review pass, D5): the MLX fit gate admits on the per-phase
    /// ALLOCATOR envelope, and one might expect that level — unlike the active counter — to
    /// carry the wired weight set. It does not, in the conditioning phase: the allocator
    /// conditioning levels are 1.26 MB (flux2_dev q4/q8), 44.1 KB (qwen_image bf16/q4/q8) and
    /// 2.27 GB (z_image_turbo q4), so the allocator-basis residues are, in GB
    /// (cond / denoise / decode): flux2_dev q4 −33.60 / +7.31 / +55.20; flux2_dev q8 −55.00 /
    /// +10.26 / +58.15; qwen_image bf16 −57.73 / +5.83 / +5.47; qwen_image q4 −28.39 / +2.92 /
    /// +1.65; qwen_image q8 −38.58 / +2.84 / +1.62; z_image_turbo q4 −3.64 / +6.27 / +8.40.
    /// Denoise and decode come back positive on the allocator basis, conditioning never does, so
    /// no packaged MLX anchor is in the law's domain on either basis and the law keeps
    /// decomposing the active peak — a switch to the allocator basis would move nothing. Both
    /// bases are asserted below so a re-captured anchor that reports its weights lands here.
    #[test]
    fn the_packaged_mlx_anchors_are_outside_the_laws_domain() {
        let packaged: Vec<&MemoryAnchor> = store()
            .anchors
            .iter()
            .filter(|anchor| anchor.backend == AnchorBackend::Mlx && anchor.geometry.frames == 1)
            .collect();
        assert!(
            packaged.len() >= 6,
            "the packaged store carries the six MLX image anchors, found {}",
            packaged.len()
        );
        for anchor in packaged {
            assert!(
                !anchor.measured_regime.staged,
                "{}: eager resident",
                anchor.id
            );
            let resident_set = packaged_tier_bytes(&anchor.model_id, &anchor.tier);
            // The split does not matter to a resident-regime decomposition (every phase is
            // decomposed against the sum); the sum is the manifest's.
            let components = ComponentBytes {
                conditioning: resident_set,
                transformer: 0,
                decoder: 0,
            };
            assert!(
                anchor.phase_active_peak_bytes.conditioning < resident_set / 2,
                "{}: conditioning active {} is not below the {resident_set} resident set it claims",
                anchor.id,
                anchor.phase_active_peak_bytes.conditioning
            );
            // …and so is the ALLOCATOR level of the same phase (D5): every packaged row carries
            // a per-phase allocator decomposition, and its conditioning level is a fraction of
            // the resident set too, while its denoise and decode levels sit above it.
            let allocators = anchor
                .phase_allocator_envelope_bytes
                .unwrap_or_else(|| panic!("{}: carries a per-phase allocator envelope", anchor.id));
            assert!(
                allocators.conditioning < resident_set / 2,
                "{}: conditioning allocator {} is not below the {resident_set} resident set",
                anchor.id,
                allocators.conditioning
            );
            assert!(
                allocators.denoise > resident_set && allocators.decode > resident_set,
                "{}: the denoise/decode allocator levels {}/{} must exceed the resident set — if \
                 this fails the census in the doc comment above has moved",
                anchor.id,
                allocators.denoise,
                allocators.decode
            );
            // A resident request with every bounded rung engaged (so neither the regime guard
            // nor a staged phase priced at a zero component refuses) is refused by the domain
            // guard alone…
            let request = at(
                anchor.geometry.width,
                anchor.geometry.height,
                RequestRegime {
                    staged: false,
                    ..FULLY_ENGAGED
                },
            );
            assert!(
                anchor
                    .derive_phase_peaks(&request, components, ArchitectureFacts::default())
                    .is_none(),
                "{}: an out-of-domain anchor must refuse, not clamp",
                anchor.id
            );
            // …and lifting every phase peak to at least the resident set is exactly what makes
            // it derivable again, so the refusal is the domain guard and nothing else. (The
            // conditioning phase is below on every packaged row; flux2_dev q4 and qwen_image
            // bf16 report a denoise active peak below their resident set as well.)
            let mut lifted = anchor.clone();
            let peaks = &mut lifted.phase_active_peak_bytes;
            peaks.conditioning = peaks.conditioning.max(resident_set);
            peaks.denoise = peaks.denoise.max(resident_set);
            peaks.decode = peaks.decode.max(resident_set);
            assert!(
                lifted
                    .derive_phase_peaks(&request, components, ArchitectureFacts::default())
                    .is_some(),
                "{}: with every phase peak at or above the resident set the anchor derives",
                anchor.id
            );
        }
    }

    /// A synthetic eager, resident, unbounded MLX anchor at 1024x1024 whose counters saw its
    /// weights: every phase peak is the whole component set plus a stated activation residue,
    /// and the allocator level sits a stated envelope above each. Identity fields are borrowed
    /// from the packaged flux2_dev q8 row (the lane entry point checks lane, load shape and
    /// regime, not the id); the peaks are NOT the record's.
    fn in_domain_resident_mlx_anchor(components: ComponentBytes) -> MemoryAnchor {
        let residues = AnchorPhaseBytes {
            conditioning: 500_000_000,
            denoise: 6_000_000_000,
            decode: 12_000_000_000,
        };
        let total = components.total();
        let mut anchor = flux2_mlx_anchor("q8").clone();
        anchor.id = "synthetic:mlx:resident:in-domain".to_owned();
        anchor.underived_reason = None;
        anchor.geometry = AnchorGeometry {
            width: 1024,
            height: 1024,
            frames: 1,
            fps: None,
        };
        anchor.phase_active_peak_bytes = AnchorPhaseBytes {
            conditioning: total + residues.conditioning,
            denoise: total + residues.denoise,
            decode: total + residues.decode,
        };
        anchor.phase_allocator_envelope_bytes = Some(AnchorPhaseBytes {
            conditioning: total + residues.conditioning + 100_000_000,
            denoise: total + residues.denoise + 1_000_000_000,
            decode: total + residues.decode + 2_000_000_000,
        });
        anchor.overall_allocator_envelope_bytes = total + residues.decode + 2_000_000_000;
        anchor
    }

    /// The rung ordering AC 2 asked of an MLX resident anchor, on an anchor inside the law's
    /// domain (the packaged ones are not — see the test above): the windowed rung prices below
    /// the staged rung, with every residue it scales asserted positive so the ordering is not
    /// vacuous, and the resident derivation at the anchor geometry reproduces the anchor.
    #[test]
    fn a_resident_mlx_anchor_whose_counters_saw_its_weights_prices_the_ladder_in_order() {
        // Qwen-Image bf16 component bytes: the tier's packaged download split at the
        // Qwen2.5-VL-7B text encoder's bf16 size (16.6 GB) and the VAE's (~254 MB).
        let total = packaged_tier_bytes("qwen_image", "bf16");
        let components = ComponentBytes {
            conditioning: 16_600_000_000,
            transformer: total - 16_600_000_000 - 253_806_592,
            decoder: 253_806_592,
        };
        // Qwen-Image: 24 heads of 128, 60 blocks, patch 2 over the 16-channel x8 VAE, bf16.
        let facts = ArchitectureFacts {
            attention_heads: Some(24),
            transformer_blocks: Some(60),
            ..Z_IMAGE_FACTS
        };
        let anchor = in_domain_resident_mlx_anchor(components);
        assert!(!anchor.measured_regime.staged);
        assert_eq!(anchor.load_shape, AnchorLoadShape::EagerMaterialization);
        for (phase, peak) in [
            ("conditioning", anchor.phase_active_peak_bytes.conditioning),
            ("denoise", anchor.phase_active_peak_bytes.denoise),
            ("decode", anchor.phase_active_peak_bytes.decode),
        ] {
            assert!(
                peak > components.total(),
                "{phase}: the residue the law scales must be positive, peak {peak} against {}",
                components.total()
            );
        }
        let rung_2 = anchor
            .derive_phase_peaks(&at(1024, 1024, RequestRegime::staged()), components, facts)
            .expect("the resident MLX anchor prices the staged rung");
        let rung_4 = anchor
            .derive_phase_peaks(&at(1024, 1024, FULLY_ENGAGED), components, facts)
            .expect("the resident MLX anchor prices the windowed rung");
        assert!(
            rung_4.peak_bytes() < rung_2.peak_bytes(),
            "windowed {rung_4:?} must price below staged {rung_2:?}"
        );
        assert!(rung_4.denoise < rung_2.denoise && rung_4.decode < rung_2.decode);
        // The resident derivation at the anchor geometry IS the anchor: every component re-added
        // into every phase plus the positive residue, byte for byte.
        let resident = anchor
            .derive_phase_peaks(
                &at(1024, 1024, RequestRegime::resident()),
                components,
                facts,
            )
            .expect("resident");
        assert_eq!(
            resident.conditioning,
            anchor.phase_active_peak_bytes.conditioning
        );
        assert_eq!(resident.denoise, anchor.phase_active_peak_bytes.denoise);
        assert_eq!(resident.decode, anchor.phase_active_peak_bytes.decode);
        assert!(rung_2.peak_bytes() <= resident.peak_bytes());
        // And the same anchor with one phase pushed below its resident set is refused whole.
        let mut below = anchor.clone();
        below.phase_active_peak_bytes.denoise = components.total() - 1;
        assert!(below
            .derive_phase_peaks(&at(1024, 1024, FULLY_ENGAGED), components, facts)
            .is_none());
    }

    /// sc-22665 (E4): the activation half of the law is exactly the whole estimate minus the
    /// components the request's regime holds resident in that phase — nothing is re-added, and a
    /// request the law refuses hands out no residue either. Asserted on the same in-domain
    /// resident MLX anchor the rung-ordering test uses, at three regimes, so the residue's own
    /// rung ordering (chunk shrinks denoise, tile shrinks decode) is visible without the
    /// components' rung ordering masking it.
    #[test]
    fn the_activation_residues_are_the_derived_peaks_minus_the_regimes_resident_components() {
        let total = packaged_tier_bytes("qwen_image", "bf16");
        let components = ComponentBytes {
            conditioning: 16_600_000_000,
            transformer: total - 16_600_000_000 - 253_806_592,
            decoder: 253_806_592,
        };
        let facts = ArchitectureFacts {
            attention_heads: Some(24),
            transformer_blocks: Some(60),
            ..Z_IMAGE_FACTS
        };
        let anchor = in_domain_resident_mlx_anchor(components);
        let chunked = RequestRegime {
            staged: true,
            attention_chunk_scores: Some(64 * 1024 * 1024),
            ..RequestRegime::staged()
        };
        let residues = |regime| {
            let request = at(1024, 1024, regime);
            let peaks = anchor
                .derive_phase_peaks(&request, components, facts)
                .expect("the in-domain anchor prices this regime");
            let residues = anchor
                .derive_phase_activation_residues(&request, components, facts)
                .expect("…and hands out the activation half of the same estimate");
            // The difference is a COMPONENT set, never an activation byte: every phase's
            // peak-minus-residue is a sum of the three component figures.
            for (phase, peak, residue) in [
                ("conditioning", peaks.conditioning, residues.conditioning),
                ("denoise", peaks.denoise, residues.denoise),
                ("decode", peaks.decode, residues.decode),
            ] {
                let resident = peak
                    .checked_sub(residue)
                    .unwrap_or_else(|| panic!("{phase}: the residue cannot exceed the peak"));
                assert!(
                    resident <= components.total(),
                    "{phase}: {resident} resident bytes exceed the whole component set"
                );
            }
            residues
        };
        let rung_2 = residues(RequestRegime::staged());
        let rung_3 = residues(chunked);
        let rung_4 = residues(FULLY_ENGAGED);
        assert_eq!(
            rung_2.conditioning, rung_4.conditioning,
            "no rung below bounds the conditioning phase"
        );
        assert!(
            rung_3.denoise < rung_2.denoise,
            "the attention chunk must shrink the denoise residue: {rung_3:?} vs {rung_2:?}"
        );
        assert_eq!(rung_3.decode, rung_2.decode, "rung 3 does not tile decode");
        assert!(
            rung_4.decode < rung_3.decode,
            "the decode tile must shrink the decode residue: {rung_4:?} vs {rung_3:?}"
        );
        assert_eq!(rung_4.denoise, rung_3.denoise);
        // A refused request hands out no residue either.
        let mut below = anchor.clone();
        below.phase_active_peak_bytes.denoise = components.total() - 1;
        assert!(below
            .derive_phase_activation_residues(&at(1024, 1024, FULLY_ENGAGED), components, facts)
            .is_none());
    }

    /// AC 3: `ArchitectureFacts::default()` leaves every residue unscaled, so the estimate is
    /// never smaller than with full facts — for the whole set and for each fact dropped alone.
    #[test]
    fn missing_facts_leave_residues_unscaled_and_never_shrink_the_estimate() {
        let request = at(1024, 1024, FULLY_ENGAGED);
        let anchor = z_image_q4_anchor();
        let full = z_image_at(&request);
        let none = anchor
            .derive_phase_peaks(
                &request,
                Z_IMAGE_Q4_COMPONENTS,
                ArchitectureFacts::default(),
            )
            .expect("no facts still derives");
        assert!(none.conditioning >= full.conditioning);
        assert!(none.denoise >= full.denoise);
        assert!(none.decode >= full.decode);
        // With no facts the window and the chunk cannot scale anything: denoise is the anchor's
        // own measured denoise (whole transformer + whole residue), i.e. the staged rung's.
        assert_eq!(none.denoise, anchor.phase_active_peak_bytes.denoise);
        assert!(none.denoise > full.denoise);
        // The decode floor needs the activation width (the blender's output accumulator is in
        // that dtype), so without facts the whole decode residue stays unscaled: decode is the
        // anchor's own measured decode, i.e. the staged rung's.
        assert_eq!(none.decode, anchor.phase_active_peak_bytes.decode);
        assert!(none.decode > full.decode);

        // Dropping any ONE fact never shrinks the estimate below the full-facts one.
        type FactDrop = fn(&mut ArchitectureFacts);
        let drops: [(&str, FactDrop); 8] = [
            ("attention_heads", |facts| facts.attention_heads = None),
            ("head_dim", |facts| facts.head_dim = None),
            ("transformer_blocks", |facts| {
                facts.transformer_blocks = None
            }),
            ("patch_size", |facts| facts.patch_size = None),
            ("latent_channels", |facts| facts.latent_channels = None),
            ("vae_spatial_scale", |facts| facts.vae_spatial_scale = None),
            ("vae_temporal_scale", |facts| {
                facts.vae_temporal_scale = None
            }),
            ("activation_dtype_width", |facts| {
                facts.activation_dtype_width = None
            }),
        ];
        for (label, drop) in drops {
            let mut facts = Z_IMAGE_FACTS;
            drop(&mut facts);
            let partial = anchor
                .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, facts)
                .unwrap_or_else(|| panic!("without {label} the law must still derive"));
            for (phase, partial_bytes, full_bytes) in [
                ("conditioning", partial.conditioning, full.conditioning),
                ("denoise", partial.denoise, full.denoise),
                ("decode", partial.decode, full.decode),
            ] {
                assert!(
                    partial_bytes >= full_bytes,
                    "without {label}, {phase} {partial_bytes} fell below the full-facts \
                     {full_bytes}"
                );
            }
        }
        // …and the facts that carry a ratio actually move it (so the assertions above are not
        // vacuous): blocks drive the window, heads/width/patch/scale drive the score split, and
        // the activation width alone drives the decode floor.
        for (label, drop) in drops {
            let mut facts = Z_IMAGE_FACTS;
            drop(&mut facts);
            let partial = anchor
                .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, facts)
                .expect("derives");
            let moves_denoise = matches!(
                label,
                "attention_heads"
                    | "transformer_blocks"
                    | "patch_size"
                    | "vae_spatial_scale"
                    | "activation_dtype_width"
            );
            assert_eq!(
                partial.denoise > full.denoise,
                moves_denoise,
                "{label}: dropping it {} change the denoise estimate",
                if moves_denoise { "must" } else { "must not" }
            );
            let moves_decode = label == "activation_dtype_width";
            assert_eq!(
                partial.decode > full.decode,
                moves_decode,
                "{label}: dropping it {} change the decode estimate",
                if moves_decode { "must" } else { "must not" }
            );
        }
    }

    // -------------------------------------------------------------------------------------
    // Image derivation law — the retained corpora as falsifiers.
    // -------------------------------------------------------------------------------------

    /// The Krea candle five-rung corpus derived from its own staged record: every rung's derived
    /// admission peak brackets the measured one, per phase where the composition matches the
    /// law's, inside a documented tightness budget — and the deeper rungs are priced strictly
    /// below the staged rung, so the ratios are known to have bitten.
    #[test]
    fn candle_anchor_derivation_brackets_every_retained_candle_measurement() {
        /// Deeper-rung tightness. The binding case is the retained window-1 rung at 2.14x its
        /// measured 3.996 GB peak, and the phase that binds is DECODE: the law scales Krea's
        /// decode residue by the larger of the two decode chunk models — the (512 − 128) x 1024
        /// host-transfer band, 3/8 of the image — because it cannot know that Krea tiles its VAE
        /// on device at 512² (1/4 of the image; see `DecodeTile::chunk_pixels`), and prices that
        /// rung's decode at 8.55 GB against a measured 3.83 GB. The next rungs sit at 1.18x and
        /// 1.02x. Forgetting to subtract the components would put the window-1 rung above 5x.
        const DEEPER_RUNG_TIGHTNESS: f64 = 2.5;
        let anchor = krea_candle_anchor();
        let corpus = krea_candle_corpus();
        assert!(
            corpus.len() >= 5,
            "the retained candle corpus must span the five rungs, found {}",
            corpus.len()
        );
        let staged = anchor
            .derive_phase_peaks(
                &at(
                    anchor.geometry.width,
                    anchor.geometry.height,
                    RequestRegime::staged(),
                ),
                KREA_Q4_COMPONENTS,
                KREA_FACTS,
            )
            .expect("the staged rung derives");
        let mut bracketed = 0usize;
        for record in &corpus {
            let derived = anchor
                .derive_phase_peaks(
                    &at(record.width, record.height, record.regime),
                    KREA_Q4_COMPONENTS,
                    KREA_FACTS,
                )
                .unwrap_or_else(|| panic!("{} must be derivable", record.id));
            let peak = derived.peak_bytes();
            assert!(
                peak >= record.peak(),
                "{} ({:?}): derived peak {peak} under-predicts the measured {}",
                record.id,
                record.engaged,
                record.peak()
            );
            assert!(
                derived.conditioning >= record.conditioning,
                "{} conditioning: derived {} under the measured {}",
                record.id,
                derived.conditioning,
                record.conditioning
            );
            assert!(
                derived.denoise >= record.denoise,
                "{} denoise: derived {} under the measured {}",
                record.id,
                derived.denoise,
                record.denoise
            );
            // Decode per phase only where the composition is the law's: Krea's staged path keeps
            // the DiT resident through decode (the retained resident-minus-staged decode delta is
            // the text encoder alone), so its tiled decode phases carry ~8.6 GB of transformer
            // the VAE-only decode composition does not price. The admission PEAK still brackets
            // them above; the decode phase itself is asserted where the anchor's composition
            // holds.
            if record.regime.decode_tile.is_none() {
                assert!(
                    derived.decode >= record.decode,
                    "{} decode: derived {} under the measured {}",
                    record.id,
                    derived.decode,
                    record.decode
                );
            }
            let ratio = peak as f64 / record.peak() as f64;
            let budget = if record.regime == RequestRegime::staged() {
                1.0
            } else {
                DEEPER_RUNG_TIGHTNESS
            };
            assert!(
                ratio <= budget,
                "{} ({:?}): derived peak {peak} is {ratio:.3}x the measured {}, over the {budget}x \
                 budget",
                record.id,
                record.engaged,
                record.peak()
            );
            if record.regime != RequestRegime::staged() && record.regime.staged {
                assert!(
                    peak < staged.peak_bytes(),
                    "{} ({:?}): a deeper rung must price below the staged rung {}",
                    record.id,
                    record.engaged,
                    staged.peak_bytes()
                );
            }
            bracketed += 1;
        }
        assert!(
            bracketed >= 5,
            "all five retained candle compositions must be bracketed, got {bracketed}"
        );
    }

    /// The candle lane's entry point is the same law: at the anchor's geometry the staged
    /// derivation equals the anchor, and it keeps the shallow-anchor guards its consumers were
    /// written against — a resident request, a deeper-measured anchor, another lane, an
    /// underived row are refused there, not by the law.
    #[test]
    fn the_candle_entry_point_keeps_its_guards_while_the_law_prices_what_it_refuses() {
        let anchor = krea_candle_anchor();
        let request = AnchorImageDeriveRequest {
            width: 1024,
            height: 1024,
            staged_residency: true,
        };
        let derived = anchor
            .derive_image_phase_peaks(request, KREA_Q4_COMPONENTS)
            .expect("the shallow staged anchor prices its own geometry");
        assert_eq!(
            derived,
            AnchorDerivedPhases {
                conditioning: anchor.phase_active_peak_bytes.conditioning,
                denoise: anchor.phase_active_peak_bytes.denoise,
                decode: anchor.phase_active_peak_bytes.decode,
            }
        );
        // The entry point refuses the resident composition; the law prices it, above staged.
        assert!(anchor
            .derive_image_phase_peaks(
                AnchorImageDeriveRequest {
                    staged_residency: false,
                    ..request
                },
                KREA_Q4_COMPONENTS,
            )
            .is_none());
        let resident = anchor
            .derive_phase_peaks(
                &at(1024, 1024, RequestRegime::resident()),
                KREA_Q4_COMPONENTS,
                KREA_FACTS,
            )
            .expect("the law prices the resident composition");
        assert!(resident.peak_bytes() > derived.peak_bytes());
        let measured_resident = krea_candle_corpus()
            .into_iter()
            .find(|record| record.engaged == ["resident"])
            .expect("the retained resident capture");
        assert!(resident.conditioning >= measured_resident.conditioning);
        assert!(resident.denoise >= measured_resident.denoise);
        assert!(resident.decode >= measured_resident.decode);

        for (label, mutate) in [
            (
                "a deeper measured regime",
                Box::new(|anchor: &mut MemoryAnchor| anchor.measured_regime.decode_tiled = true)
                    as Box<dyn Fn(&mut MemoryAnchor)>,
            ),
            (
                "a resident measured regime",
                Box::new(|anchor: &mut MemoryAnchor| anchor.measured_regime.staged = false),
            ),
            (
                "the other backend lane",
                Box::new(|anchor: &mut MemoryAnchor| anchor.backend = AnchorBackend::Mlx),
            ),
            (
                "a stated underived reason",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.underived_reason = Some("validation-only".to_owned());
                }),
            ),
        ] {
            let mut mutated = anchor.clone();
            mutate(&mut mutated);
            assert!(
                mutated
                    .derive_image_phase_peaks(request, KREA_Q4_COMPONENTS)
                    .is_none(),
                "the candle entry point must refuse an anchor carrying {label}"
            );
        }
    }

    /// The identity and regime guards of the law itself, each mutated individually onto an
    /// otherwise-derivable anchor.
    #[test]
    fn the_law_refuses_video_identities_and_unbounded_requests_of_bounded_anchors() {
        let anchor = z_image_q4_anchor();
        let request = at(1024, 1024, FULLY_ENGAGED);
        assert!(anchor
            .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
            .is_some());
        for (label, mutate) in [
            (
                "a video pipeline variant",
                Box::new(|anchor: &mut MemoryAnchor| {
                    anchor.transformer_variant = Some(Ltx25TransformerVariant::Dev);
                }) as Box<dyn Fn(&mut MemoryAnchor)>,
            ),
            (
                "a video decoder",
                Box::new(|anchor: &mut MemoryAnchor| anchor.decoder = Some(Ltx25Decoder::Conv)),
            ),
            (
                "a multi-frame measured geometry",
                Box::new(|anchor: &mut MemoryAnchor| anchor.geometry.frames = 145),
            ),
        ] {
            let mut mutated = anchor.clone();
            mutate(&mut mutated);
            assert!(
                mutated
                    .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
                    .is_none(),
                "the law must refuse an anchor carrying {label}"
            );
        }
        // The packaged LTX video anchors are refused for their axes, and the video law refuses
        // the image anchors in turn.
        let ltx = store()
            .anchor_for(
                "ltx_2_5",
                AnchorBackend::Mlx,
                "bf16",
                Ltx25TransformerVariant::Dev,
                Ltx25Decoder::DiffVae,
            )
            .expect("the LTX bf16 dev/diffvae anchor");
        assert!(ltx
            .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
            .is_none());
        assert!(krea_candle_anchor()
            .derive_video_phase_peaks(plain_request(1024, 1024, 1))
            .is_none());

        // A phase the anchor measured BOUNDED cannot price the same phase unbounded…
        for (label, measured, unbounded) in [
            (
                "tiled",
                AnchorMeasuredRegime {
                    decode_tiled: true,
                    ..anchor.measured_regime
                },
                RequestRegime {
                    decode_tile: None,
                    ..FULLY_ENGAGED
                },
            ),
            (
                "chunked",
                AnchorMeasuredRegime {
                    attention_chunked: true,
                    ..anchor.measured_regime
                },
                RequestRegime {
                    attention_chunk_scores: None,
                    ..FULLY_ENGAGED
                },
            ),
            (
                "windowed",
                AnchorMeasuredRegime {
                    transformer_windowed: true,
                    ..anchor.measured_regime
                },
                RequestRegime {
                    transformer_window: None,
                    ..FULLY_ENGAGED
                },
            ),
        ] {
            let mut bounded = anchor.clone();
            bounded.measured_regime = measured;
            assert!(
                bounded
                    .derive_phase_peaks(
                        &at(1024, 1024, unbounded),
                        Z_IMAGE_Q4_COMPONENTS,
                        Z_IMAGE_FACTS
                    )
                    .is_none(),
                "a {label}-measured anchor must refuse the unbounded request"
            );
            // …while the correspondingly bounded request reuses the bounded residue UNSCALED:
            // it prices at or above what the unbounded anchor's scaled ratio would give.
            let reused = bounded
                .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
                .unwrap_or_else(|| panic!("a {label}-measured anchor prices the bounded request"));
            let scaled = anchor
                .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
                .expect("the unbounded anchor prices the bounded request");
            assert!(reused.peak_bytes() >= scaled.peak_bytes());
        }

        // Degenerate inputs are refused, not clamped.
        for (label, request) in [
            ("a zero width", at(0, 1024, FULLY_ENGAGED)),
            (
                "a zero batch",
                ImageDeriveRequest {
                    batch: 0,
                    ..request
                },
            ),
            (
                "a zero tile edge",
                at(
                    1024,
                    1024,
                    RequestRegime {
                        decode_tile: Some(DecodeTile {
                            edge: 0,
                            overlap: 0,
                        }),
                        ..FULLY_ENGAGED
                    },
                ),
            ),
            (
                "a tile overlap not below its edge (the engines refuse the tuple too)",
                at(
                    1024,
                    1024,
                    RequestRegime {
                        decode_tile: Some(DecodeTile {
                            edge: 512,
                            overlap: 512,
                        }),
                        ..FULLY_ENGAGED
                    },
                ),
            ),
            (
                "a zero chunk",
                at(
                    1024,
                    1024,
                    RequestRegime {
                        attention_chunk_scores: Some(0),
                        ..FULLY_ENGAGED
                    },
                ),
            ),
            (
                "a zero window",
                at(
                    1024,
                    1024,
                    RequestRegime {
                        transformer_window: Some(0),
                        ..FULLY_ENGAGED
                    },
                ),
            ),
        ] {
            assert!(
                anchor
                    .derive_phase_peaks(&request, Z_IMAGE_Q4_COMPONENTS, Z_IMAGE_FACTS)
                    .is_none(),
                "{label} must be refused"
            );
        }
    }

    /// The residue split never goes negative: an anchor whose denoise residue is smaller than
    /// the full score tensor its facts imply caps the subtraction at the residue, so the chunked
    /// estimate is the resident set plus the chunk and nothing less, and the unchunked estimate
    /// at the anchor geometry re-prices the full tensor above the measured peak rather than below.
    #[test]
    fn the_score_subtraction_is_capped_at_the_residue() {
        let anchor = z_image_q4_anchor();
        // 300 heads make the 1024x1024 score tensor ~12.7 GB, far above the 4.58 GB residue.
        let facts = ArchitectureFacts {
            attention_heads: Some(300),
            ..Z_IMAGE_FACTS
        };
        let chunked = anchor
            .derive_phase_peaks(
                &at(
                    1024,
                    1024,
                    RequestRegime {
                        attention_chunk_scores: Some(64 * 1024 * 1024),
                        ..RequestRegime::staged()
                    },
                ),
                Z_IMAGE_Q4_COMPONENTS,
                facts,
            )
            .expect("derives");
        assert_eq!(
            chunked.denoise,
            Z_IMAGE_Q4_COMPONENTS.transformer + 64 * 1024 * 1024 * 2
        );
        let unchunked = anchor
            .derive_phase_peaks(
                &at(1024, 1024, RequestRegime::staged()),
                Z_IMAGE_Q4_COMPONENTS,
                facts,
            )
            .expect("derives");
        assert!(unchunked.denoise > anchor.phase_active_peak_bytes.denoise);
    }

    /// A chunk budget larger than the whole score tensor is capped at the tensor: the chunked
    /// rung can never price above the unchunked one. At 256x256 the Z-Image tensor is
    /// 30 x 1536² x 2 B = 141.6 MB, so a 1 Gi-score chunk (2 GiB) is capped to it; at 1024x1024
    /// the 64 Mi chunk sits below the 1.27 GB tensor and must bite by exactly the difference.
    #[test]
    fn the_chunk_term_is_capped_at_the_unchunked_score_tensor() {
        let anchor = z_image_q4_anchor();
        let chunked = |width: u32, chunk: u64| {
            anchor
                .derive_phase_peaks(
                    &at(
                        width,
                        width,
                        RequestRegime {
                            attention_chunk_scores: Some(chunk),
                            ..RequestRegime::staged()
                        },
                    ),
                    Z_IMAGE_Q4_COMPONENTS,
                    Z_IMAGE_FACTS,
                )
                .expect("derives")
        };
        let unchunked = |width: u32| {
            anchor
                .derive_phase_peaks(
                    &at(width, width, RequestRegime::staged()),
                    Z_IMAGE_Q4_COMPONENTS,
                    Z_IMAGE_FACTS,
                )
                .expect("derives")
        };
        // 256x256: 32² + 512 = 1536 tokens, tensor 30 x 1536² x 2 B = 141,557,760 B; a 1 Gi
        // chunk (2 GiB of scores) is far above it and is capped, so chunked == unchunked.
        let oversized = 1024 * 1024 * 1024;
        assert!(oversized * 2 > 141_557_760);
        assert_eq!(chunked(256, oversized).denoise, unchunked(256).denoise);
        // 1024x1024: the 64 Mi chunk is below the 1,274,019,840 B tensor and bites by exactly
        // the difference.
        let chunk_bytes = 64 * 1024 * 1024 * 2;
        assert!(chunk_bytes < 1_274_019_840);
        assert_eq!(
            unchunked(1024).denoise - chunked(1024, 64 * 1024 * 1024).denoise,
            1_274_019_840 - chunk_bytes
        );
    }

    /// The conditioning residue scales with the prompt's conditioning tokens over the default
    /// — and never below 1.0, since the records do not state the prompt length they were
    /// measured with. The joint sequence length moves the denoise estimate with it.
    #[test]
    fn conditioning_tokens_scale_the_conditioning_residue_and_never_credit_a_short_prompt() {
        let anchor = z_image_q4_anchor();
        let with_tokens = |tokens: Option<u32>| {
            anchor
                .derive_phase_peaks(
                    &ImageDeriveRequest {
                        conditioning_tokens: tokens,
                        ..at(1024, 1024, RequestRegime::staged())
                    },
                    Z_IMAGE_Q4_COMPONENTS,
                    Z_IMAGE_FACTS,
                )
                .expect("derives")
        };
        let default = with_tokens(None);
        assert_eq!(default, with_tokens(Some(DEFAULT_CONDITIONING_TOKENS)));
        let residue =
            anchor.phase_active_peak_bytes.conditioning - Z_IMAGE_Q4_COMPONENTS.conditioning;
        assert!(residue > 0);
        // Twice the tokens: twice the residue on top of the text encoder, and a longer joint
        // sequence for denoise.
        let doubled = with_tokens(Some(2 * DEFAULT_CONDITIONING_TOKENS));
        assert_eq!(
            doubled.conditioning,
            Z_IMAGE_Q4_COMPONENTS.conditioning + 2 * residue
        );
        assert!(doubled.denoise > default.denoise);
        // Half the tokens: no credit anywhere — the request's count is floored at the default
        // on every side it enters, so the estimate is the default's in every phase.
        let halved = with_tokens(Some(DEFAULT_CONDITIONING_TOKENS / 2));
        assert_eq!(halved, default);
    }

    /// The fixture component bytes are not free literals: each set sums to its tier's packaged
    /// download (`estimatedSizeBytes` in the builtin manifest) within 2%, so a retyped figure
    /// or a re-cut tier lands here.
    #[test]
    fn the_fixture_component_bytes_sum_to_their_packaged_tier_sizes() {
        for (label, components, model, tier) in [
            (
                "Z-Image-Turbo q4",
                Z_IMAGE_Q4_COMPONENTS,
                "z_image_turbo",
                "q4",
            ),
            ("FLUX.2 [dev] q4", flux2_components("q4"), "flux2_dev", "q4"),
            ("FLUX.2 [dev] q8", flux2_components("q8"), "flux2_dev", "q8"),
        ] {
            let packaged = packaged_tier_bytes(model, tier);
            let sum = components.total();
            let drift = (sum as f64 - packaged as f64).abs() / packaged as f64;
            assert!(
                drift <= 0.02,
                "{label}: components sum to {sum} against the packaged {packaged} ({drift:.4})"
            );
            assert!(components.conditioning > 0 && components.transformer > 0);
            assert!(components.decoder > 0);
        }
        assert_eq!(packaged_tier_bytes("z_image_turbo", "q4"), 5_909_406_487);
    }

    /// Batch multiplies the activation residues (and the score tensor), never the weights.
    #[test]
    fn batch_scales_the_residues_and_not_the_components() {
        let anchor = z_image_q4_anchor();
        let request = at(1024, 1024, RequestRegime::staged());
        let one = z_image_at(&request);
        let two = z_image_at(&ImageDeriveRequest {
            batch: 2,
            ..request
        });
        assert!(two.conditioning > one.conditioning);
        assert!(two.denoise > one.denoise);
        assert!(two.decode > one.decode);
        let components = Z_IMAGE_Q4_COMPONENTS;
        let residue = |peak: u64, resident: u64| peak - resident;
        let peaks = anchor.phase_active_peak_bytes;
        assert_eq!(
            two.conditioning,
            components.conditioning + 2 * residue(peaks.conditioning, components.conditioning)
        );
        assert_eq!(
            two.decode,
            components.decoder + 2 * residue(peaks.decode, components.decoder)
        );
        assert_eq!(
            two.denoise,
            components.transformer + 2 * residue(peaks.denoise, components.transformer)
        );
    }

    // -------------------------------------------------------------------------------------
    // MLX image lane entry point.
    // -------------------------------------------------------------------------------------

    fn flux2_mlx_anchor(tier: &str) -> &'static MemoryAnchor {
        store()
            .image_anchor_for("flux2_dev", AnchorBackend::Mlx, tier)
            .unwrap_or_else(|| panic!("the flux2_dev {tier} MLX anchor is packaged"))
    }

    /// FLUX.2 [dev] MLX component bytes per tier: the tier's packaged download
    /// (`estimatedSizeBytes` in the builtin manifest — 33.6 GB q4, 55.0 GB q8) split at the
    /// 24B Mistral text encoder's 4-/8-bit size and the VAE. Bound to the manifest by
    /// `the_fixture_component_bytes_sum_to_their_packaged_tier_sizes` (within 2%).
    fn flux2_components(tier: &str) -> ComponentBytes {
        match tier {
            "q4" => ComponentBytes {
                conditioning: 13_400_000_000,
                transformer: 19_850_000_000,
                decoder: 350_000_000,
            },
            "q8" => ComponentBytes {
                conditioning: 25_400_000_000,
                transformer: 29_250_000_000,
                decoder: 350_000_000,
            },
            other => panic!("no flux2 components for {other}"),
        }
    }

    /// The MLX entry point's guards, each mutated individually on an otherwise-derivable anchor
    /// (an in-domain synthetic one — the packaged rows are refused by the law's domain guard,
    /// asserted here too), and its area scaling in both directions.
    #[test]
    fn the_mlx_entry_point_keeps_its_guards_and_tracks_output_area() {
        let request = AnchorMlxImageDeriveRequest {
            width: 1024,
            height: 1024,
        };
        let components = flux2_components("q4");
        let in_domain = in_domain_resident_mlx_anchor(components);
        assert!(in_domain
            .derive_mlx_image_phase_peaks(request, components)
            .is_some());
        for (label, mutate) in [
            (
                "the candle lane",
                Box::new(|anchor: &mut MemoryAnchor| anchor.backend = AnchorBackend::Candle)
                    as Box<dyn Fn(&mut MemoryAnchor)>,
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
            let mut mutated = in_domain.clone();
            mutate(&mut mutated);
            assert!(
                mutated
                    .derive_mlx_image_phase_peaks(request, components)
                    .is_none(),
                "the MLX entry point must refuse an anchor carrying {label}"
            );
        }
        // Every packaged MLX image anchor is refused at the entry point: the flux2 rows by the
        // law's domain guard (their conditioning counters saw no weights), the single-geometry
        // rows by their stated underived reason AND the domain guard.
        for (model, tier) in [
            ("flux2_dev", "q4"),
            ("flux2_dev", "q8"),
            ("qwen_image", "bf16"),
            ("z_image_turbo", "q4"),
        ] {
            let anchor = store()
                .image_anchor_for(model, AnchorBackend::Mlx, tier)
                .unwrap_or_else(|| panic!("{model} {tier} anchor"));
            let components = ComponentBytes {
                conditioning: packaged_tier_bytes(model, tier),
                transformer: 0,
                decoder: 0,
            };
            assert!(anchor
                .derive_mlx_image_phase_peaks(request, components)
                .is_none());
            assert!(anchor
                .derive_phase_peaks(
                    &at(1024, 1024, RequestRegime::resident()),
                    components,
                    ArchitectureFacts::default()
                )
                .is_none());
        }
        // Output-area tracking through the entry point, on the in-domain anchor: the allocator
        // envelope is carried unscaled below the anchor geometry and pixel-scaled above it.
        let anchor = in_domain_resident_mlx_anchor(flux2_components("q8"));
        let at = |width: u32, height: u32| {
            anchor
                .derive_mlx_image_phase_peaks(
                    AnchorMlxImageDeriveRequest { width, height },
                    flux2_components("q8"),
                )
                .unwrap_or_else(|| panic!("{width}x{height} must be derivable"))
        };
        assert!(at(512, 512).peak_bytes() < at(768, 768).peak_bytes());
        assert!(at(768, 768).peak_bytes() < at(1024, 1024).peak_bytes());
        assert!(at(1024, 1024).peak_bytes() < at(1536, 1536).peak_bytes());
        assert_eq!(
            at(1024, 1024).peak_bytes(),
            anchor.overall_allocator_envelope_bytes,
            "at the anchor geometry the entry point reproduces the anchor's envelope"
        );
        assert_eq!(at(1344, 768).peak_bytes(), at(768, 1344).peak_bytes());
        assert!(anchor
            .derive_mlx_image_phase_peaks(
                AnchorMlxImageDeriveRequest {
                    width: 0,
                    height: 512,
                },
                flux2_components("q8"),
            )
            .is_none());
    }

    /// An anchor row may carry component bytes; a stated set that sums to zero is refused at
    /// load, and the field round-trips through the store.
    #[test]
    fn component_bytes_round_trip_through_the_store_and_a_zero_set_is_refused() {
        let mut doctored: serde_json::Value =
            serde_json::from_str(PACKAGED_MEMORY_ANCHORS).expect("packaged store parses");
        let index = store()
            .anchors
            .iter()
            .position(|anchor| anchor.model_id == "krea_2_turbo")
            .expect("the krea anchor");
        doctored["anchors"][index]["componentBytes"] = serde_json::json!({
            "conditioning": KREA_Q4_COMPONENTS.conditioning,
            "transformer": KREA_Q4_COMPONENTS.transformer,
            "decoder": KREA_Q4_COMPONENTS.decoder,
        });
        let loaded = load_memory_anchors(&doctored.to_string())
            .expect("a store carrying component bytes loads");
        assert_eq!(
            loaded.anchors[index].component_bytes,
            Some(KREA_Q4_COMPONENTS)
        );
        let serialized = serde_json::to_string(&loaded).expect("serializes");
        assert!(serialized.contains("\"componentBytes\""));
        assert_eq!(
            load_memory_anchors(&serialized)
                .expect("round-trips")
                .anchors[index]
                .component_bytes,
            Some(KREA_Q4_COMPONENTS)
        );
        // Every packaged row states none today; `None` never serializes.
        assert!(store()
            .anchors
            .iter()
            .all(|anchor| anchor.component_bytes.is_none()));
        assert!(!PACKAGED_MEMORY_ANCHORS.contains("componentBytes"));

        doctored["anchors"][index]["componentBytes"] = serde_json::json!({
            "conditioning": 0,
            "transformer": 0,
            "decoder": 0,
        });
        let error = load_memory_anchors(&doctored.to_string())
            .expect_err("all-zero component bytes must be rejected");
        assert!(error.contains("sum to zero"), "{error}");
    }

    /// An underived VIDEO anchor prices nothing: the video law fits, so it honours the field.
    #[test]
    fn an_underived_video_anchor_prices_nothing() {
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
        assert!(video
            .derive_video_phase_peaks(plain_request(768, 512, 121))
            .is_some());
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
}
