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

#[path = "sdxl_cuda_memory_endpoints.rs"]
pub mod sdxl_cuda_memory_endpoints;

pub const MEMORY_ANCHOR_SCHEMA_VERSION: u32 = 1;

/// The checked-in anchor store.
pub const PACKAGED_MEMORY_ANCHORS: &str = include_str!("../../../config/memory-anchors.json");

/// Immutable retained evidence the anchors were extracted from. Every anchor's `source.path` must
/// name one of these files, whose bytes are compiled in so the handshake cannot be bypassed by
/// editing the file on disk.
///
/// EVERY retained corpus is packaged (sc-22666, epic 22657 E5). Packaging used to be a per-model
/// opt-in because the candle image lane priced a cell with per-pixel slopes fitted on Krea Turbo,
/// so anchoring another model from a freshly committed corpus would have repriced it with borrowed
/// slopes. Since sc-22663 the image lanes fit nothing — the law decomposes each anchor's OWN
/// measured peaks against the contract's component bytes — so there is no slope left to borrow and
/// a retained corpus that is not compiled in is a defect rather than a deferral.
/// `scripts/extract-memory-anchors.mjs` fails when it walks a corpus that could anchor a catalog
/// cell but is absent from this list, so the packaging can never silently lapse again.
///
/// Keep sorted by path; `include_str!` paths are relative to this crate's `src/`.
const PACKAGED_MEMORY_ANCHOR_SOURCES: &[(&str, &str)] = &[
    (
        "docs/calibration/sc-15859/z-image-turbo-bf16-candle-anchor.json",
        include_str!("../../../docs/calibration/sc-15859/z-image-turbo-bf16-candle-anchor.json"),
    ),
    (
        "docs/calibration/sc-15859/z-image-turbo-q4-candle-anchor.json",
        include_str!("../../../docs/calibration/sc-15859/z-image-turbo-q4-candle-anchor.json"),
    ),
    (
        "docs/calibration/sc-15859/z-image-turbo-q8-candle-anchor.json",
        include_str!("../../../docs/calibration/sc-15859/z-image-turbo-q8-candle-anchor.json"),
    ),
    (
        "docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json",
        include_str!("../../../docs/calibration/sc-18791/ltx25-mlx-evidence.seed.json"),
    ),
    (
        "docs/calibration/sc-22738/bernini-bf16-mlx-exceeded-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/bernini-bf16-mlx-exceeded-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/bernini-image-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/bernini-image-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/bernini-image-bf16-mlx-exceeded-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/bernini-image-bf16-mlx-exceeded-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/bernini-image-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/bernini-image-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/bernini-image-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/bernini-image-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/bernini-image-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/bernini-image-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/bernini-image-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/bernini-image-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/bernini-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/bernini-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/bernini-q4-mlx-exceeded-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/bernini-q4-mlx-exceeded-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/bernini-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/bernini-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/bernini-q8-mlx-exceeded-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/bernini-q8-mlx-exceeded-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-base-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-base-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-base-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-base-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-base-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-base-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-base-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-base-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-base-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-base-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-base-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-base-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-flash-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-flash-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-flash-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-flash-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-flash-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-flash-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-flash-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-flash-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-flash-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-flash-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-flash-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-flash-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-hd-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-hd-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-hd-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-hd-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-hd-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-hd-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-hd-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-hd-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-hd-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-hd-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/chroma1-hd-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/chroma1-hd-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-dev-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-dev-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-dev-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-dev-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-dev-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-dev-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-dev-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-dev-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-dev-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-dev-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-dev-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-dev-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-schnell-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-schnell-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-schnell-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-schnell-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-schnell-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-schnell-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-schnell-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-schnell-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-schnell-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-schnell-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux-schnell-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux-schnell-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-dev-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-dev-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-dev-bf16-mlx-exceeded-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/flux2-dev-bf16-mlx-exceeded-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/flux2-dev-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-dev-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-dev-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-dev-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-dev-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-dev-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-dev-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-dev-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-klein-9b-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-klein-9b-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-kv-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/flux2-klein-9b-kv-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-kv-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-klein-9b-kv-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-kv-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/flux2-klein-9b-kv-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-kv-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-klein-9b-kv-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-kv-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/flux2-klein-9b-kv-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-kv-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-klein-9b-kv-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-klein-9b-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-klein-9b-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-klein-9b-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/flux2-klein-9b-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/flux2-klein-9b-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ideogram-4-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ideogram-4-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ideogram-4-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ideogram-4-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ideogram-4-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ideogram-4-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-turbo-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/ideogram-4-turbo-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-turbo-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ideogram-4-turbo-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-turbo-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ideogram-4-turbo-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-turbo-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ideogram-4-turbo-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-turbo-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ideogram-4-turbo-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ideogram-4-turbo-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ideogram-4-turbo-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/illustrious-xl-v1-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/illustrious-xl-v1-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/illustrious-xl-v1-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/illustrious-xl-v1-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/illustrious-xl-v1-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/illustrious-xl-v1-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/illustrious-xl-v2-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/illustrious-xl-v2-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/illustrious-xl-v2-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/illustrious-xl-v2-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/illustrious-xl-v2-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/illustrious-xl-v2-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/instantid-realvisxl-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/instantid-realvisxl-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/kolors-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/kolors-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/kolors-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/kolors-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/kolors-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/kolors-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-2-raw-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/krea-2-raw-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-2-raw-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/krea-2-raw-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-2-raw-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/krea-2-raw-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-2-raw-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/krea-2-raw-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-2-raw-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/krea-2-raw-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-2-raw-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/krea-2-raw-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-2-turbo-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/krea-2-turbo-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-2-turbo-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/krea-2-turbo-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-2-turbo-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/krea-2-turbo-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-2-turbo-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/krea-2-turbo-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-2-turbo-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/krea-2-turbo-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/krea-realtime-14b-bf16-mlx-exceeded-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/krea-realtime-14b-bf16-mlx-exceeded-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/krea-realtime-14b-q4-mlx-exceeded-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/krea-realtime-14b-q4-mlx-exceeded-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/krea-realtime-14b-q8-mlx-exceeded-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/krea-realtime-14b-q8-mlx-exceeded-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/lens-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/lens-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/lens-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/lens-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/lens-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/lens-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/lens-turbo-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-turbo-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/lens-turbo-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-turbo-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/lens-turbo-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-turbo-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/lens-turbo-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-turbo-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/lens-turbo-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-turbo-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/lens-turbo-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/lens-turbo-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ltx-2-3-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ltx-2-3-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ltx-2-3-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ltx-2-3-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ltx-2-3-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ltx-2-3-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ltx-2-3-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ltx-2-3-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ltx-2-3-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ltx-2-3-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ltx-2-5-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ltx-2-5-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ltx-2-5-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ltx-2-5-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ltx-2-5-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ltx-2-5-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ltx-2-5-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ltx-2-5-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ltx-2-5-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ltx-2-5-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/ltx-2-5-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/ltx-2-5-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-base-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-base-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-base-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-base-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-base-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-base-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-base-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-base-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-base-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-base-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-base-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-base-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-base-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/mage-flow-edit-base-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-base-bf16-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/mage-flow-edit-base-bf16-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-base-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/mage-flow-edit-base-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-base-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-edit-base-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-base-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/mage-flow-edit-base-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-base-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-edit-base-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-edit-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-edit-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-edit-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-edit-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-edit-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-edit-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-turbo-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/mage-flow-edit-turbo-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-turbo-bf16-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/mage-flow-edit-turbo-bf16-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-turbo-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/mage-flow-edit-turbo-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-turbo-q4-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/mage-flow-edit-turbo-q4-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-turbo-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/mage-flow-edit-turbo-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-edit-turbo-q8-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/mage-flow-edit-turbo-q8-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-turbo-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/mage-flow-turbo-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-turbo-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-turbo-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-turbo-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-turbo-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-turbo-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-turbo-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-turbo-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-turbo-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/mage-flow-turbo-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/mage-flow-turbo-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/minimax-h3-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/minimax-h3-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/minimax-h3-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/minimax-h3-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/minimax-h3-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/minimax-h3-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/minimax-h3-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/minimax-h3-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/minimax-h3-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/minimax-h3-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/minimax-h3-ref-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/minimax-h3-ref-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/minimax-h3-ref-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/minimax-h3-ref-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/minimax-h3-ref-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/minimax-h3-ref-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/minimax-h3-ref-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/minimax-h3-ref-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/minimax-h3-ref-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/minimax-h3-ref-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/pulid-flux-dev-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/pulid-flux-dev-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/pulid-flux-dev-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/pulid-flux-dev-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/pulid-flux-dev-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/pulid-flux-dev-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/qwen-image-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/qwen-image-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-edit-2511-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/qwen-image-edit-2511-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-edit-2511-bf16-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/qwen-image-edit-2511-bf16-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-edit-2511-lightning-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/qwen-image-edit-2511-lightning-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-edit-2511-lightning-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/qwen-image-edit-2511-lightning-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-edit-2511-lightning-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/qwen-image-edit-2511-lightning-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-edit-2511-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/qwen-image-edit-2511-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-edit-2511-q4-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/qwen-image-edit-2511-q4-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-edit-2511-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/qwen-image-edit-2511-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-edit-2511-q8-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/qwen-image-edit-2511-q8-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/qwen-image-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/qwen-image-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/qwen-image-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/qwen-image-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/qwen-image-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/realvisxl-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/realvisxl-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/realvisxl-lightning-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/realvisxl-lightning-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/realvisxl-lightning-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/realvisxl-lightning-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/realvisxl-lightning-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/realvisxl-lightning-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/realvisxl-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/realvisxl-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/realvisxl-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/realvisxl-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sana-1600m-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sana-1600m-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sana-1600m-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sana-1600m-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sana-1600m-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sana-1600m-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sana-1600m-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sana-1600m-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sana-sprint-1600m-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sana-sprint-1600m-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sana-sprint-1600m-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sana-sprint-1600m-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sana-sprint-1600m-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sana-sprint-1600m-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sana-sprint-1600m-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sana-sprint-1600m-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/scail2-14b-bf16-mlx-exceeded-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/scail2-14b-bf16-mlx-exceeded-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/scail2-14b-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/scail2-14b-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/scail2-14b-q4-mlx-exceeded-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/scail2-14b-q4-mlx-exceeded-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/scail2-14b-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/scail2-14b-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/scail2-14b-q8-mlx-exceeded-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/scail2-14b-q8-mlx-exceeded-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-large-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-large-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-large-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-large-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-large-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-large-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-turbo-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sd3-5-large-turbo-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-turbo-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-large-turbo-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-turbo-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sd3-5-large-turbo-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-turbo-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-large-turbo-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-turbo-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sd3-5-large-turbo-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-large-turbo-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-large-turbo-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-medium-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-medium-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-medium-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-medium-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-medium-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-medium-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-medium-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-medium-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-medium-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-medium-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sd3-5-medium-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sd3-5-medium-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sdxl-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sdxl-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sdxl-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sdxl-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sdxl-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sdxl-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sensenova-u1-8b-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-fast-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-fast-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-fast-bf16-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-fast-bf16-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-fast-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-fast-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-fast-q4-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-fast-q4-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-fast-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-fast-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-fast-q8-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-fast-q8-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-bf16-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-bf16-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-bf16-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-bf16-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-q4-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-q4-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-q8-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-fast-q8-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-q4-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-q4-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-q8-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v2-q8-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-bf16-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-bf16-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-bf16-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-bf16-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-q4-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-q4-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-q8-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-fast-q8-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-q4-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-q4-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-q8-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/sensenova-u1-8b-infographic-v3-q8-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sensenova-u1-8b-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sensenova-u1-8b-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sensenova-u1-8b-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/sensenova-u1-8b-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/sensenova-u1-8b-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/wan-2-2-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/wan-2-2-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/wan-2-2-bf16-mlx-exceeded-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/wan-2-2-bf16-mlx-exceeded-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/wan-2-2-i2v-14b-bf16-mlx-exceeded-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/wan-2-2-i2v-14b-bf16-mlx-exceeded-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/wan-2-2-i2v-14b-q4-mlx-exceeded-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/wan-2-2-i2v-14b-q4-mlx-exceeded-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/wan-2-2-i2v-14b-q8-mlx-exceeded-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/wan-2-2-i2v-14b-q8-mlx-exceeded-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/wan-2-2-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/wan-2-2-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/wan-2-2-q4-mlx-exceeded-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/wan-2-2-q4-mlx-exceeded-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/wan-2-2-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/wan-2-2-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/wan-2-2-q8-mlx-exceeded-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/wan-2-2-q8-mlx-exceeded-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/wan-2-2-t2v-14b-bf16-mlx-exceeded-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-22738/wan-2-2-t2v-14b-bf16-mlx-exceeded-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-22738/z-image-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-edit-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-edit-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-edit-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-edit-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-edit-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-edit-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-edit-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-edit-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-edit-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-edit-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-edit-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-edit-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-q4-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-q4-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-q8-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-q8-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-turbo-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-turbo-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-22738/z-image-turbo-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-22738/z-image-turbo-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-24114/qwen-image-2-1-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-24114/qwen-image-2-1-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-24114/qwen-image-2-1-q4-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-24114/qwen-image-2-1-q4-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-24114/qwen-image-2-1-q8-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-24114/qwen-image-2-1-q8-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-24217/supra2-img-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-24217/supra2-img-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-24217/supra2-img-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-24217/supra2-img-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-25690/iris-3b-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-25690/iris-3b-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-25690/iris-3b-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-25690/iris-3b-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-25690/iris-3b-depth-bf16-candle-evidence.json",
        include_str!("../../../docs/calibration/sc-25690/iris-3b-depth-bf16-candle-evidence.json"),
    ),
    (
        "docs/calibration/sc-25690/iris-3b-depth-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-25690/iris-3b-depth-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-25690/iris-3b-restore-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-25690/iris-3b-restore-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-25690/iris-3b-restore-bf16-mlx-evidence.json",
        include_str!("../../../docs/calibration/sc-25690/iris-3b-restore-bf16-mlx-evidence.json"),
    ),
    (
        "docs/calibration/sc-25909/qwen-image-2-1-turbo-bf16-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-25909/qwen-image-2-1-turbo-bf16-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-25909/qwen-image-2-1-turbo-bf16-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-25909/qwen-image-2-1-turbo-bf16-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-25909/qwen-image-2-1-turbo-q4-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-25909/qwen-image-2-1-turbo-q4-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-25909/qwen-image-2-1-turbo-q4-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-25909/qwen-image-2-1-turbo-q4-mlx-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-25909/qwen-image-2-1-turbo-q8-candle-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-25909/qwen-image-2-1-turbo-q8-candle-evidence.json"
        ),
    ),
    (
        "docs/calibration/sc-25909/qwen-image-2-1-turbo-q8-mlx-evidence.json",
        include_str!(
            "../../../docs/calibration/sc-25909/qwen-image-2-1-turbo-q8-mlx-evidence.json"
        ),
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
        "docs/generated/ltx-mlx-video-sc-18808.json",
        include_str!("../../../docs/generated/ltx-mlx-video-sc-18808.json"),
    ),
    (
        "docs/generated/memory-calibration-evidence.json",
        include_str!("../../../docs/generated/memory-calibration-evidence.json"),
    ),
    (
        "docs/generated/qwen-candle-five-rung-sc-15817.json",
        include_str!("../../../docs/generated/qwen-candle-five-rung-sc-15817.json"),
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
/// Retained MLX same-cell active-peak recapture spread. Estimated video peaks include this
/// uncertainty once; reclaimable cache is excluded. The worker's policy audits the same value
/// against the retained evidence corpus and asserts equality with this shared derivation input.
pub const MLX_ACTIVE_RECAPTURE_SPREAD: f64 = 0.1260183508475594;

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
// Five of the six packaged MLX image anchors are outside the domain today — their
// conditioning-phase active peak is a fraction of the eager resident set they claim, and so is
// their conditioning-phase ALLOCATOR level (the quantity the MLX fit gate admits on), so
// decomposing the allocator envelope instead would not bring one in (see
// `the_packaged_mlx_anchors_are_outside_the_laws_domain` for both censuses). The cause was the
// adapter's capture shape, not the counters (sc-22667, epic 22657 D3): MLX residency is
// request-scoped even under eager materialization, so a window opened on a freshly loaded
// generator measured a cold first request that was still materializing its weights. The
// `z_image_turbo` q4 anchor was re-captured with the window opened above its materialized
// resident set and is in domain (`the_recaptured_z_image_mlx_anchor_is_inside_the_laws_domain`);
// the others enter the domain the same way, one re-capture each.
//
// The measured phase level the law subtracts from is [`MemoryAnchor::phase_active_peak_bytes`],
// the store's byte-bound decomposition. MLX allocator cache is reclaimable and is not added
// to the active-memory prediction. The admission caller carries recapture uncertainty separately.
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

/// Validation-only tightness of sibling+delta active estimates, including recapture uncertainty.
/// The retained leave-one-out bound is 23.88% above its active peak (bf16 distilled/conv
/// 1280x704, imc-4e1b3a02a6ced3434824). Round up to the next percentage point. This constrains
/// the estimate in tests; it is never an additional admission allowance.
pub const ANCHOR_DELTA_VALIDATION_TIGHTNESS_BUDGET: f64 = 0.24;

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
    /// Measured LOWER BOUNDS from captures a physical-footprint hard stop ended (sc-22738, epic
    /// 22723). This is the store's THIRD entry kind and the only one whose evidence is a render
    /// that did NOT complete: see [`ExceededBound`]. `default` so a store written before the
    /// migration still parses.
    #[serde(default)]
    pub exceeded_bounds: Vec<ExceededBound>,
    /// Sampled accounting observations, NEVER input to the legacy residency/allocator laws.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observed_operating_envelopes: Vec<ObservedOperatingEnvelope>,
}

/// Exact executed recipe identity. Hashes bind actual native structs, not requested profiles.
/// Matching this identity does not turn an accounting observation into a capacity guarantee.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ObservedExecutionRecipe {
    pub schema_version: u32,
    pub backend: AnchorBackend,
    pub provider: String,
    pub tier: String,
    pub actual_load_spec_sha256: String,
    pub actual_generation_request_sha256: String,
    pub invocation_request_sha256: String,
    pub measured_strategy_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnedProcessAccountingPhase {
    pub dedicated_peak_bytes: u64,
    pub shared_peak_bytes: u64,
    pub total_committed_peak_bytes: u64,
    pub samples: u64,
    pub first_sample_ms: u64,
    pub last_sample_ms: u64,
    pub max_gap_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnedProcessAccountingPhases {
    pub load: OwnedProcessAccountingPhase,
    pub conditioning: OwnedProcessAccountingPhase,
    pub denoise: OwnedProcessAccountingPhase,
    pub decode: OwnedProcessAccountingPhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_upsampling: Option<OwnedProcessAccountingPhase>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnedProcessAccountingIdentity {
    pub pid: u32,
    pub started_at: String,
    pub executable: String,
    pub luid: String,
    pub physical_gpu_id: String,
    pub gpu_uuid: String,
    pub pci_bus_id: String,
    pub logical_cuda_ordinal: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnedProcessAccountingSampling {
    pub interval_ms: u64,
    pub max_allowed_gap_ms: u64,
    pub max_allowed_read_skew_ms: u64,
    pub max_observed_read_skew_ms: u64,
    pub lossless_peak_claim: bool,
}

/// Unit serializes as JSON null; non-null counters cannot deserialize into unavailable metrics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UnavailableAccountingMetrics {
    pub physical_resident_bytes: (),
    pub active_bytes: (),
    pub allocator_bytes: (),
}

/// Windows process accounting is not physical residency, allocator activity, or an upper bound
/// on all CUDA demand. Independent peaks are not additive. Shared accounting is not proof of
/// physical spill. No method on this type yields a legacy active/allocator peak or a fit verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnedProcessAccountingEvidence {
    pub schema_version: u32,
    pub kind: String,
    pub source: String,
    pub semantics: String,
    pub attribution: String,
    pub physical_residency_claim: bool,
    pub allocator_activity_claim: bool,
    pub device_total_delta_attribution_claim: bool,
    pub exclusive_throughput_claim: bool,
    pub sampling: OwnedProcessAccountingSampling,
    pub identity: OwnedProcessAccountingIdentity,
    pub physical_nodes: Vec<u32>,
    pub phases: OwnedProcessAccountingPhases,
    pub overall: OwnedProcessAccountingPhase,
    pub unavailable_metrics: UnavailableAccountingMetrics,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AccountingObservationSource {
    pub path: String,
    pub sha256: String,
    pub record_id: String,
}

/// A source-bound observed execution, kept separate from analytic residency anchors. Exact recipe
/// lookup is observation-only: changing geometry, artifacts, request or native choices is outside
/// its domain. Fit/admission additionally requires independently observed operating conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ObservedOperatingEnvelope {
    pub id: String,
    pub model_id: String,
    pub recipe: ObservedExecutionRecipe,
    pub accounting: OwnedProcessAccountingEvidence,
    pub operating_conditions: ObservedOperatingConditions,
    pub source: AccountingObservationSource,
}

/// Admission-instant foreign/headroom snapshot; not a phase minimum or a reservation guarantee.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ObservedOperatingConditions {
    pub schema_version: u32,
    pub sharing_mode: String,
    pub physical_gpu_id: String,
    pub gpu_uuid: String,
    pub pci_bus_id: String,
    pub observed_at: String,
    pub total_device_bytes: u64,
    pub free_device_bytes: u64,
    pub foreign_dedicated_bytes: u64,
    pub foreign_process_count: u64,
    pub foreign_max_process_bytes: u64,
    pub foreign_process_limit_bytes: u64,
    pub foreign_total_limit_bytes: u64,
    pub observation_scope: String,
    pub fit_authorization: bool,
    pub exclusive_throughput_claim: bool,
    pub physical_residency_claim: bool,
}

fn accounting_timestamp_valid(value: &str) -> bool {
    let bytes = value.as_bytes();
    if !(22..=30).contains(&bytes.len())
        || !bytes.iter().enumerate().all(|(index, byte)| match index {
            4 | 7 => *byte == b'-',
            10 => *byte == b'T',
            13 | 16 => *byte == b':',
            19 => *byte == b'.',
            end if end == bytes.len() - 1 => *byte == b'Z',
            _ => byte.is_ascii_digit(),
        })
    {
        return false;
    }
    let number = |start, end| value[start..end].parse::<u32>().unwrap_or(0);
    let year = number(0, 4);
    let month = number(5, 7);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        _ => 0,
    };
    year > 0
        && number(8, 10) > 0
        && number(8, 10) <= days
        && number(11, 13) < 24
        && number(14, 16) < 60
        && number(17, 19) < 60
}

impl OwnedProcessAccountingEvidence {
    pub fn validate(&self) -> Result<(), String> {
        let identity = &self.identity;
        let luid_parts: Vec<_> = identity.luid.split('_').collect();
        let luid_valid = luid_parts.len() == 2
            && luid_parts.iter().all(|part| {
                part.len() == 10
                    && part.starts_with("0x")
                    && part[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
            });
        if self.schema_version != 1
            || self.kind != "flux3_owned_process_memory_accounting_v1"
            || self.source != "Windows GPU Process Memory performance counters"
            || self.semantics != "sampled per-process dedicated/shared/total-committed accounting; not physical residency or CUDA active/reserved allocator bytes"
            || self.attribution != "own PID/start-time/executable and CUDA adapter LUID; includes any cross-process-shared allocations charged to this process"
            || self.physical_residency_claim || self.allocator_activity_claim
            || self.device_total_delta_attribution_claim || self.exclusive_throughput_claim
            || self.sampling.interval_ms != 200 || self.sampling.max_allowed_gap_ms != 1500
            || self.sampling.max_allowed_read_skew_ms != 250 || self.sampling.max_observed_read_skew_ms > 250 || self.sampling.lossless_peak_claim
            || identity.pid == 0 || identity.logical_cuda_ordinal != 0 || !luid_valid
            || !matches!(identity.physical_gpu_id.as_str(), "0" | "1")
            || !identity.gpu_uuid.strip_prefix("GPU-").is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_hexdigit() || byte == b'-'))
            || identity.pci_bus_id.len() != 16 || !identity.pci_bus_id.bytes().enumerate().all(|(index, byte)| match index { 8 | 11 => byte == b':', 14 => byte == b'.', 15 => (b'0'..=b'7').contains(&byte), _ => byte.is_ascii_hexdigit() })
            || !accounting_timestamp_valid(&identity.started_at)
            || [&identity.executable, &identity.physical_gpu_id, &identity.gpu_uuid, &identity.pci_bus_id].iter().any(|value| value.trim().is_empty())
            || self.physical_nodes.is_empty()
            || self.physical_nodes.iter().collect::<std::collections::BTreeSet<_>>().len() != self.physical_nodes.len()
        {
            return Err("invalid owned-process accounting semantics, sampling or identity".into());
        }
        let overall = &self.overall;
        let mut phases = vec![&self.phases.load];
        if let Some(prompt) = &self.phases.prompt_upsampling {
            phases.push(prompt);
        }
        phases.extend([
            &self.phases.conditioning,
            &self.phases.denoise,
            &self.phases.decode,
        ]);
        for phase in std::iter::once(overall).chain(phases.iter().copied()) {
            let span = phase.last_sample_ms.saturating_sub(phase.first_sample_ms);
            if phase.samples < 2
                || phase.last_sample_ms <= phase.first_sample_ms
                || phase.max_gap_ms == 0
                || phase.max_gap_ms > 1500
                || phase.max_gap_ms > span
                || span > phase.max_gap_ms.saturating_mul(phase.samples - 1)
            {
                return Err("invalid owned-process phase coverage or sample gap".into());
            }
            if phase.first_sample_ms < overall.first_sample_ms
                || phase.last_sample_ms > overall.last_sample_ms
                || phase.dedicated_peak_bytes > overall.dedicated_peak_bytes
                || phase.shared_peak_bytes > overall.shared_peak_bytes
                || phase.total_committed_peak_bytes > overall.total_committed_peak_bytes
                || phase.samples > overall.samples
            {
                return Err("owned-process overall does not cover its phases".into());
            }
        }
        if phases
            .windows(2)
            .any(|pair| pair[0].last_sample_ms != pair[1].first_sample_ms)
        {
            return Err(
                "owned-process phase boundaries lack a synchronized shared endpoint".into(),
            );
        }
        Ok(())
    }
}

/// A MEASURED LOWER BOUND on one cell's peak: the request reached `observed_footprint_bytes` of
/// kernel physical footprint on a host of `host_memory_bytes` and was terminated by the capture
/// guard before it finished (sc-22738, epic 22723 E4/E5).
///
/// WHY THIS IS EVIDENCE AND NOT A DROPPED CAPTURE. Before this row existed a footprint hard stop
/// stamped nothing at all: `measure-memory-catalog.mjs` recorded `capture_failed`, the store kept
/// no trace, and production kept admitting the very request that had just been killed — the
/// `bernini:bf16:mlx` 848x480x49 render that ran 72 minutes and reached 97,147,294,328 bytes
/// before the guard stopped it mid-decode. The run learned the single most decision-relevant fact
/// about the cell (its peak is AT LEAST that), and the pipeline threw it away.
///
/// WHAT IT CLAIMS, AND WHAT IT DELIBERATELY DOES NOT. It claims one inequality — `peak >=
/// observed_footprint_bytes` at this geometry — and nothing else. It is NOT a phase decomposition,
/// NOT a point a law may fit, and NOT a substitute for a [`MemoryAnchor`]: it prices no estimate,
/// widens no envelope and enters no derivation. Its ONLY consumer is the pre-load refusal
/// ([`ExceededBound::required_bytes`]), which is why every field it carries is either identity or
/// the inequality's own terms.
///
/// IT IS RETIRED ONLY BY RE-MEASUREMENT (sc-22738). A LATER completed capture of the same cell —
/// same identity axes, geometry covering the bound's, on a host no larger than the one that was
/// stopped — supersedes it: `scripts/extract-memory-anchors.mjs` (`retainExceededBounds`) drops
/// the bound from the store and the anchor stands in its place. A later stop at the same geometry
/// replaces the earlier bound rather than sitting beside it. Stops at other geometries all stay,
/// each the strongest true inequality for its own point. Nothing else lifts a bound.
///
/// IT BINDS REGARDLESS OF CURRENCY (sc-22738). The bound carries the same
/// [`AnchorSource::loader_closure_digest`] an anchor does, and for the same reason: so the probe
/// tooling can tell when the loader that produced it has moved and the cell is worth stopping at
/// again — the runner (`measure-memory-catalog.mjs`) then classifies the cell runnable and the
/// MLX memory adapter's pre-load refusal lets it through to the guarded capture, whose outcome
/// supersedes the bound as above. The runtime never reads that key. A bound whose closure has
/// moved refuses exactly what it refused the day it was measured, because the alternative — a
/// shared-engine fix silently re-admitting the very request a host was already unable to finish —
/// is the failure the bound exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExceededBound {
    pub id: String,
    pub model_id: String,
    pub model_family: String,
    pub route: String,
    pub provider: String,
    pub backend: AnchorBackend,
    pub tier: String,
    pub transformer_variant: Option<Ltx25TransformerVariant>,
    pub decoder: Option<Ltx25Decoder>,
    pub mode: String,
    pub overlay: Option<String>,
    pub reference_count: u32,
    pub load_shape: AnchorLoadShape,
    pub geometry: AnchorGeometry,
    /// The kernel `phys_footprint` of the guarded process group at the sample that tripped the
    /// hard stop. The cell's true peak is at or above this: the render was still allocating.
    pub observed_footprint_bytes: u64,
    /// The guard's hard-stop ceiling for that run, carried so a reader can see how far past it the
    /// group had already gone in one sampling interval.
    pub ceiling_bytes: u64,
    /// Whole-host memory of the capture host. Recorded because the inequality is only interesting
    /// relative to a budget, and a bound measured on a smaller host must not read as a claim about
    /// a larger one.
    pub host_memory_bytes: u64,
    /// The guard's own hard-stop reason, verbatim.
    pub reason: String,
    pub source: AnchorSource,
}

impl ExceededBound {
    /// Whether a request at `(width, height, frames)` is at or above the geometry this bound was
    /// measured at, on every axis. Monotonicity is the whole claim: a render that already needed
    /// at least X bytes at this geometry cannot need less at a strictly larger one. A request
    /// SMALLER on any axis is outside the bound and takes no refusal from it.
    pub const fn covers_geometry(&self, width: u32, height: u32, frames: u32) -> bool {
        width >= self.geometry.width
            && height >= self.geometry.height
            && frames >= self.geometry.frames
    }

    /// What a host must be able to give this request before it may be admitted: the measured lower
    /// bound plus the lane's activation headroom.
    ///
    /// The headroom term is not padding. The footprint reading is a LOWER bound taken at the
    /// instant the guard fired — for the Bernini capture, mid-VAE-decode on a curve that had just
    /// climbed 18 GiB in under a second — so admitting at exactly the bound would admit a render
    /// already known to want more. Charging the lane's ordinary activation allowance on top is the
    /// same term every floor candidate already carries, applied to a peak the measurement proved
    /// rather than to one an estimate guessed.
    pub const fn required_bytes(&self, headroom_bytes: u64) -> u64 {
        self.observed_footprint_bytes.saturating_add(headroom_bytes)
    }

    /// Whether this bound refuses a host — the whole decision, stated once so production and the
    /// capture-side mirror cannot drift.
    ///
    /// TWO disjuncts, and the FIRST is the load-bearing one:
    ///
    /// * **A host no larger than the one that failed.** This is what the measurement literally
    ///   says: a machine of `host_memory_bytes` ran this render and could not finish it. Nothing
    ///   about a same-sized machine makes it likelier to succeed, so it is refused outright. This
    ///   disjunct is the reason the fix bites at all — the Bernini stop was taken on the same
    ///   128 GiB Mac that would otherwise queue the render again, and a footprint-versus-budget
    ///   comparison alone would have re-admitted it: 90.5 GiB plus the lane's 16 GiB allowance is
    ///   106.5 GiB, which an idle 128 GiB host appears to have.
    /// * **A larger host that cannot currently carry the requirement.** A bigger machine MIGHT
    ///   finish this render, so it is graded on what it can actually offer right now rather than
    ///   refused on the strength of a smaller host's failure.
    ///
    /// `host_total_bytes` is the budget's own physical ceiling, which on the MLX lane can be the
    /// allocator's limit rather than `hw.memsize` — i.e. at or below the figure a bound records.
    /// The comparison is deliberately left inclusive-and-conservative there: erring towards
    /// refusal on a host of ambiguous size is the direction that does not take the machine down.
    pub const fn refuses_host(
        &self,
        host_total_bytes: u64,
        effective_bytes: u64,
        headroom_bytes: u64,
    ) -> bool {
        host_total_bytes <= self.host_memory_bytes
            || effective_bytes < self.required_bytes(headroom_bytes)
    }
}

/// The identity a request presents to [`MemoryAnchorStore::exceeded_bound_for`]. Every conjunct is
/// an axis a [`MemoryAnchor`] lookup also keys on, for the same reason: a bound measured on one
/// provider, mode or pipeline cell says nothing about another's.
#[derive(Debug, Clone, Copy)]
pub struct ExceededBoundQuery<'a> {
    pub model_id: &'a str,
    /// The two axes only a CATALOG resolution can state. Production always states both — the
    /// worker has resolved the catalog family and the engine route before it asks — and the
    /// capture-side mirror in the memory adapter states neither, because a capture plan names a
    /// provider and a model id and there is no catalog inside the adapter binary to resolve them
    /// from.
    ///
    /// `None` is therefore "this caller cannot state the axis", not a wildcard for anyone: both
    /// are redundant with axes already graded — the catalog family is a function of `model_id`,
    /// and the route of `provider` — so omitting them narrows nothing that `model_id`, `provider`,
    /// `backend` and `tier` have not already pinned. A caller that CAN state them must, so a
    /// mis-resolved route cannot borrow another cell's bound.
    pub model_family: Option<&'a str>,
    pub route: Option<&'a str>,
    pub provider: &'a str,
    pub backend: AnchorBackend,
    pub tier: &'a str,
    pub transformer_variant: Option<Ltx25TransformerVariant>,
    pub decoder: Option<Ltx25Decoder>,
    pub mode: &'a str,
    pub overlay: Option<&'a str>,
    pub reference_count: u32,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
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
    /// The model's backend block publishes a `memoryStrategyContract` (sc-22666, epic 22657 E5):
    /// no render is retained for the cell, but the contract's per-rung declaration plus the image
    /// law's ratios price every rung of the ladder from the manifest row, so the cell's estimate
    /// is a CONTRACT-ONLY per-rung ladder rather than one manifest scalar repeated.
    ContractEstimate,
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
    ///
    /// A RE-CAPTURE SIGNAL FOR THE PROBE TOOLING ONLY (sc-22738). `measure-memory-catalog.mjs`,
    /// `generate-memory-matrix.mjs` (`current` per anchor, `summary.staleAnchors`) and
    /// `stale-lane-report.mjs` compare this key to decide what to re-capture. Nothing in the
    /// runtime does: an anchor binds, derives and a bound refuses exactly as if this key matched,
    /// whether or not it does. There is no `is_current` on this type on purpose — a runtime
    /// function that could express "stale" is how a shared-engine fix that touches nearly every
    /// closure would silently move what a live request gets.
    pub loader_closure_digest: String,
    /// HOW the key above was derived (sc-22667). `None`: at the record's own measurement revision,
    /// the default. `Some`: at [`AnchorCurrencyAttestation::attested_revision`] instead, on the
    /// strength of a reviewed attestation in `config/anchor-currency-attestations.json` that the
    /// loader closure's diff from the measurement revision to that one is accounting-only or was
    /// witnessed unchanged by a re-measure on the same hardware. Copied into the store by
    /// `scripts/anchor-loader-closure.mjs --stamp-anchors`, never written here; carried so a
    /// current anchor cannot hide whether it is current by measurement or by attestation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub currency_attestation: Option<AnchorCurrencyAttestation>,
}

/// A reviewed statement that an anchor's evidence still describes the loader at a revision later
/// than its measurement (sc-22667, epic 22657). The repository's invalidation doctrine is
/// two-gated: a differing loader-closure digest says the SOURCE moved, and only a
/// load-or-device-path change with no behaviour witness says the MEMORY BEHAVIOUR did. This is the
/// second gate written down — `why` classifies every changed closure file, `witness` names the
/// re-measure (or states there is none and why the diff alone suffices). It is bounded: it names
/// one `attested_revision`, and a pin bump that moves the loader closure past it stales the anchor
/// again exactly as if the attestation were absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnchorCurrencyAttestation {
    /// The inference revision the cited record was measured at — must equal the record's own.
    pub measured_revision: String,
    /// The inference revision the currency key was derived at instead.
    pub attested_revision: String,
    /// When the diff was read (ISO date).
    pub attested_at: String,
    /// The story that reviewed it.
    pub story: String,
    /// `accounting-only` (every changed file prices, reports or tests; none loads or executes) or
    /// `witnessed-unchanged` (load-path files changed, and a re-measure on the same hardware
    /// reproduced the anchor).
    pub class: String,
    /// The file-by-file reading of the diff.
    pub why: String,
    /// The behaviour witness, or the stated reason none was needed.
    pub witness: String,
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
    /// The provider contract classifies `staged_residency` STRUCTURALLY NOT APPLICABLE on this
    /// anchor's lane (sc-22734), so the cell has no staged composition to be measured in.
    ///
    /// This is an architecture fact about the engine, not a property of the render: SenseNova-U1 is
    /// one fused dual-path checkpoint whose conditioning and denoise weights are interleaved in
    /// every resident layer, so there is no separable conditioning component to release and no
    /// phase boundary to stage at. The extractor writes it from the model's own manifest
    /// (`<lane>.memoryStrategyStructuralExemptions.staged_residency`, whose `evidence[].source`
    /// names the engine's `memory_strategy.rs`), never as a blanket switch.
    ///
    /// [`MemoryAnchor::derive_image_phase_peaks`] reads it to admit the ONE composition such a cell
    /// can be captured in. Its ordinary refusal of a resident candle anchor rests on the staged
    /// anchor being the SHALLOWER of two real compositions — a resident render additionally holds
    /// the text encoder through denoise and decode, which the staged intercepts do not price. Where
    /// staging is structurally impossible that asymmetry does not exist: resident IS the shallowest
    /// composition, there is no co-resident encoder above it, and the law prices it exactly as the
    /// MLX lane already prices its resident anchors.
    ///
    /// `#[serde(default)]` + `skip_serializing_if`: every packaged row predates this field, stays
    /// byte-identical, and keeps the old behaviour (`false` → the staged-only guard, unchanged).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub staged_residency_structurally_not_applicable: bool,
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
    validate_exceeded_bounds(&store)?;
    validate_observed_operating_envelopes(&store)?;
    Ok(store)
}

impl ObservedOperatingEnvelope {
    pub fn validate(&self) -> Result<(), String> {
        validate_accounting_context(
            &self.model_id,
            &self.recipe,
            &self.accounting,
            &self.operating_conditions,
        )
    }
}

pub(crate) fn validate_accounting_context(
    model_id: &str,
    recipe: &ObservedExecutionRecipe,
    accounting: &OwnedProcessAccountingEvidence,
    conditions: &ObservedOperatingConditions,
) -> Result<(), String> {
    accounting.validate()?;
    let identity = &accounting.identity;
    if !model_id.starts_with("flux3")
        || recipe.schema_version != 1
        || recipe.backend != AnchorBackend::Candle
        || !matches!(recipe.provider.as_str(), "flux3_klein" | "flux3_klein_base")
        || recipe.tier.trim().is_empty()
        || ![
            &recipe.actual_load_spec_sha256,
            &recipe.actual_generation_request_sha256,
            &recipe.invocation_request_sha256,
            &recipe.measured_strategy_sha256,
        ]
        .iter()
        .all(|hash| is_sha256(hash))
        || conditions.schema_version != 1
        || conditions.sharing_mode != "bounded-shared"
        || conditions.observation_scope != "producer-reservation-admission"
        || conditions.fit_authorization
        || conditions.exclusive_throughput_claim
        || conditions.physical_residency_claim
        || !accounting_timestamp_valid(&conditions.observed_at)
        || conditions.physical_gpu_id != identity.physical_gpu_id
        || conditions.gpu_uuid != identity.gpu_uuid
        || conditions.pci_bus_id != identity.pci_bus_id
        || conditions.total_device_bytes == 0
        || conditions.free_device_bytes == 0
        || conditions.free_device_bytes > conditions.total_device_bytes
        || conditions.foreign_dedicated_bytes >= conditions.total_device_bytes
        || conditions.foreign_process_limit_bytes != 1_073_741_824
        || conditions.foreign_total_limit_bytes != 3_221_225_472
        || conditions.foreign_dedicated_bytes > conditions.foreign_total_limit_bytes
        || conditions.foreign_max_process_bytes > conditions.foreign_process_limit_bytes
        || conditions.foreign_max_process_bytes > conditions.foreign_dedicated_bytes
        || (conditions.foreign_process_count == 0
            && (conditions.foreign_dedicated_bytes != 0
                || conditions.foreign_max_process_bytes != 0))
    {
        return Err("invalid observed recipe or admission-instant operating conditions".into());
    }
    Ok(())
}

impl MemoryAnchorStore {
    /// Exact executed-recipe observation only. This never authorizes fit, scales geometry, or
    /// enters a residency derivation. Hardware and contemporaneous conditions remain visible.
    pub fn observed_operating_envelopes_for<'a>(
        &'a self,
        model_id: &'a str,
        recipe: &'a ObservedExecutionRecipe,
    ) -> impl Iterator<Item = &'a ObservedOperatingEnvelope> + 'a {
        self.observed_operating_envelopes
            .iter()
            .filter(move |entry| entry.model_id == model_id && &entry.recipe == recipe)
    }
}

pub(crate) fn accounting_digest(domain: &str, value: &serde_json::Value) -> String {
    fn canonical(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let sorted: BTreeMap<_, _> = map
                    .iter()
                    .map(|(key, value)| (key.clone(), canonical(value)))
                    .collect();
                serde_json::Value::Object(sorted.into_iter().collect())
            }
            serde_json::Value::Array(array) => {
                serde_json::Value::Array(array.iter().map(canonical).collect())
            }
            _ => value.clone(),
        }
    }
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update(serde_json::to_vec(&canonical(value)).expect("JSON values serialize"));
    format!("{:x}", hasher.finalize())
}

fn validate_observed_operating_envelopes(store: &MemoryAnchorStore) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for entry in &store.observed_operating_envelopes {
        entry.validate()?;
        if !seen.insert(&entry.id) {
            return Err(format!("duplicate accounting observation {}", entry.id));
        }
        let (_, raw) = PACKAGED_MEMORY_ANCHOR_SOURCES
            .iter()
            .find(|(path, _)| *path == entry.source.path)
            .ok_or_else(|| {
                format!(
                    "accounting observation {} cites an unpackaged source",
                    entry.id
                )
            })?;
        if format!("{:x}", Sha256::digest(raw.as_bytes())) != entry.source.sha256 {
            return Err(format!(
                "accounting observation {} source digest mismatch",
                entry.id
            ));
        }
        let corpus: serde_json::Value =
            serde_json::from_str(raw).map_err(|error| error.to_string())?;
        let records = corpus
            .get("records")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "accounting observation source has no records".to_string())?;
        let matches: Vec<_> = records
            .iter()
            .filter(|record| {
                record.get("id").and_then(serde_json::Value::as_str)
                    == Some(entry.source.record_id.as_str())
            })
            .collect();
        if matches.len() != 1 {
            return Err("accounting observation source record is missing or ambiguous".into());
        }
        let record = matches[0];
        if record
            .get("measurementKind")
            .and_then(serde_json::Value::as_str)
            != Some("windows-owned-process-accounting-v1")
            || record.get("backend").and_then(serde_json::Value::as_str) != Some("candle")
            || record
                .get("evidenceScope")
                .and_then(serde_json::Value::as_str)
                != Some("authoritative")
            || !matches!(
                record.get("status").and_then(serde_json::Value::as_str),
                Some("complete" | "runtime_complete")
            )
            || record.get("observedMemory").is_some()
            || record
                .pointer("/target/modelId")
                .and_then(serde_json::Value::as_str)
                != Some(entry.model_id.as_str())
            || record
                .pointer("/target/provider")
                .and_then(serde_json::Value::as_str)
                != Some(entry.recipe.provider.as_str())
            || record
                .pointer("/target/tier")
                .and_then(serde_json::Value::as_str)
                != Some(entry.recipe.tier.as_str())
        {
            return Err("accounting observation source identity/kind disagrees".into());
        }
        let read = |key: &str| {
            record
                .get(key)
                .cloned()
                .ok_or_else(|| format!("accounting record lacks {key}"))
        };
        let accounting: OwnedProcessAccountingEvidence =
            serde_json::from_value(read("ownedProcessMemory")?)
                .map_err(|error| error.to_string())?;
        let recipe: ObservedExecutionRecipe =
            serde_json::from_value(read("observedExecutionRecipe")?)
                .map_err(|error| error.to_string())?;
        let conditions: ObservedOperatingConditions =
            serde_json::from_value(read("observedOperatingConditions")?)
                .map_err(|error| error.to_string())?;
        if record
            .get("observedExecutionRecipeSha256")
            .and_then(serde_json::Value::as_str)
            != Some(
                accounting_digest(
                    "flux3_observed_execution_recipe_v1\0",
                    &read("observedExecutionRecipe")?,
                )
                .as_str(),
            )
            || recipe.measured_strategy_sha256
                != accounting_digest("flux3_observed_memory_strategy_v1\0", &read("strategy")?)
        {
            return Err("accounting observation execution recipe/strategy binding changed".into());
        }
        if record
            .pointer("/diagnostics/measurements")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|measurements| {
                measurements.iter().any(|measurement| {
                    measurement
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|name| name.ends_with("DevicePeakDelta"))
                })
            })
        {
            return Err("accounting observation aliases legacy device deltas".into());
        }
        if accounting != entry.accounting
            || recipe != entry.recipe
            || conditions != entry.operating_conditions
        {
            return Err("accounting observation drifted from retained typed evidence".into());
        }
    }
    Ok(())
}

/// An attestation is a justification or it is nothing (sc-22667): both revisions must be real
/// and distinct, and the reading of the diff must be stated. What is NOT checked here is whether
/// the attested revision equals the pin — that is the currency verdict, reported by the probe
/// tooling and never a load failure, exactly as a mismatched digest is. One check for both row
/// kinds that carry an [`AnchorSource`]: an anchor and a measured lower bound are stamped by the
/// same walk (`anchor-loader-closure.mjs --stamp-anchors`) and attested from the same file.
fn validate_currency_attestation(
    kind: &str,
    id: &str,
    source: &AnchorSource,
) -> Result<(), String> {
    let Some(attestation) = &source.currency_attestation else {
        return Ok(());
    };
    for (field, revision) in [
        ("measuredRevision", &attestation.measured_revision),
        ("attestedRevision", &attestation.attested_revision),
    ] {
        if !is_revision(revision) {
            return Err(format!(
                "{kind} {id} currency attestation {field} {revision:?} is not a 40-hex revision"
            ));
        }
    }
    if attestation.measured_revision == attestation.attested_revision {
        return Err(format!(
            "{kind} {id} currency attestation attests its own measurement revision {}",
            attestation.measured_revision
        ));
    }
    for (field, value) in [
        ("attestedAt", &attestation.attested_at),
        ("story", &attestation.story),
        ("class", &attestation.class),
        ("why", &attestation.why),
        ("witness", &attestation.witness),
    ] {
        if value.trim().is_empty() {
            return Err(format!(
                "{kind} {id} currency attestation states no {field} — an attestation without its \
                 justification is a re-stamp"
            ));
        }
    }
    Ok(())
}

/// The exceeded-bound half of the store's invariants (sc-22738).
///
/// The handshake is the anchor's, narrowed to the fields the inequality is made of: the cited
/// corpus must be compiled in, hash to what the row recorded, and carry an `exceededBounds` entry
/// whose identity, geometry and byte figures are the row's. The store may not drift from the
/// retained evidence it cites, and a bound is the one row kind where drift would turn a refusal on.
fn validate_exceeded_bounds(store: &MemoryAnchorStore) -> Result<(), String> {
    let mut seen = BTreeMap::new();
    for bound in &store.exceeded_bounds {
        if let Some(previous) = seen.insert(bound.id.clone(), &bound.id) {
            return Err(format!("duplicate exceeded bound {previous}"));
        }
        validate_currency_attestation("exceeded bound", &bound.id, &bound.source)?;
        if bound.geometry.width == 0 || bound.geometry.height == 0 || bound.geometry.frames == 0 {
            return Err(format!(
                "exceeded bound {} has a degenerate geometry",
                bound.id
            ));
        }
        if bound.reason.trim().is_empty() {
            return Err(format!(
                "exceeded bound {} states no hard-stop reason — an unexplained refusal is a gap \
                 wearing a row's clothes",
                bound.id
            ));
        }
        // The guard samples and terminates at or above its ceiling, so a row claiming a footprint
        // BELOW the ceiling it cites did not come from a hard stop at all.
        if bound.ceiling_bytes == 0 || bound.observed_footprint_bytes < bound.ceiling_bytes {
            return Err(format!(
                "exceeded bound {} records footprint {} under its own hard-stop ceiling {} — that \
                 is not a run the guard stopped",
                bound.id, bound.observed_footprint_bytes, bound.ceiling_bytes
            ));
        }
        if bound.host_memory_bytes < bound.observed_footprint_bytes {
            return Err(format!(
                "exceeded bound {} records a footprint larger than the whole capture host",
                bound.id
            ));
        }
        if !is_sha256(&bound.source.loader_closure_digest) {
            return Err(format!(
                "exceeded bound {} loader closure digest {} is not a sha256",
                bound.id, bound.source.loader_closure_digest
            ));
        }
        let Some((_, source_raw)) = PACKAGED_MEMORY_ANCHOR_SOURCES
            .iter()
            .find(|(path, _)| *path == bound.source.path)
        else {
            return Err(format!(
                "exceeded bound {} cites source {} which is not a compiled retained-evidence file",
                bound.id, bound.source.path
            ));
        };
        let digest = format!("{:x}", Sha256::digest(source_raw.as_bytes()));
        if digest != bound.source.sha256 {
            return Err(format!(
                "exceeded bound {} source digest mismatch for {}: recorded {} actual {digest}",
                bound.id, bound.source.path, bound.source.sha256
            ));
        }
        let source: serde_json::Value = serde_json::from_str(source_raw).map_err(|error| {
            format!(
                "retained evidence {} does not parse: {error}",
                bound.source.path
            )
        })?;
        let entry = source
            .get("exceededBounds")
            .and_then(|entries| entries.as_array())
            .and_then(|entries| {
                entries.iter().find(|entry| {
                    entry.get("id").and_then(|id| id.as_str())
                        == Some(bound.source.record_id.as_str())
                })
            })
            .ok_or_else(|| {
                format!(
                    "exceeded bound {} cites entry {} absent from the exceededBounds of {}",
                    bound.id, bound.source.record_id, bound.source.path
                )
            })?;
        let target = entry.get("target").unwrap_or(&serde_json::Value::Null);
        let str_at = |value: &serde_json::Value, key: &str| {
            value
                .get(key)
                .and_then(|value| value.as_str())
                .map(str::to_owned)
        };
        let u64_at = |value: &serde_json::Value, key: &str| value.get(key).and_then(|v| v.as_u64());
        let entry_route = str_at(target, "route").or_else(|| str_at(target, "provider"));
        let load_shape_key = match bound.load_shape {
            AnchorLoadShape::EagerMaterialization => "eager_materialization",
            AnchorLoadShape::DeferredMaterialization => "deferred_materialization",
        };
        let geometry = target.get("geometry").unwrap_or(&serde_json::Value::Null);
        let entry_overlay = str_at(target, "overlay")
            .filter(|overlay| overlay != "none")
            .filter(|overlay| !overlay.is_empty());
        if str_at(target, "modelId").as_deref() != Some(bound.model_id.as_str())
            || str_at(target, "tier").as_deref() != Some(bound.tier.as_str())
            || str_at(entry, "backend").as_deref() != Some(bound.backend.as_key())
            || str_at(target, "mode").as_deref() != Some(bound.mode.as_str())
            || str_at(target, "provider").as_deref() != Some(bound.provider.as_str())
            || entry_route.as_deref() != Some(bound.route.as_str())
            || optional_axis_key(str_at(target, "transformerVariant").as_deref())
                != variant_key_opt(bound.transformer_variant)
            || optional_axis_key(str_at(target, "decoder").as_deref())
                != decoder_key_opt(bound.decoder)
            || str_at(entry, "loadShape").as_deref() != Some(load_shape_key)
            || str_at(entry, "calibrationFingerprint").as_deref()
                != Some(bound.source.calibration_fingerprint.as_str())
            || entry_overlay != bound.overlay
            || u64_at(entry, "referenceCount") != Some(u64::from(bound.reference_count))
        {
            return Err(format!(
                "exceeded bound {} identity disagrees with its source entry {}",
                bound.id, bound.source.record_id
            ));
        }
        if u64_at(geometry, "width") != Some(u64::from(bound.geometry.width))
            || u64_at(geometry, "height") != Some(u64::from(bound.geometry.height))
            || u64_at(geometry, "frames") != Some(u64::from(bound.geometry.frames))
            || u64_at(entry, "observedFootprintBytes") != Some(bound.observed_footprint_bytes)
            || u64_at(entry, "ceilingBytes") != Some(bound.ceiling_bytes)
            || entry
                .get("hardware")
                .and_then(|hardware| u64_at(hardware, "memoryBytes"))
                != Some(bound.host_memory_bytes)
            || str_at(entry, "reason").as_deref() != Some(bound.reason.as_str())
        {
            return Err(format!(
                "exceeded bound {} figures disagree with its source entry {} — the store may not \
                 drift from the retained evidence it cites",
                bound.id, bound.source.record_id
            ));
        }
    }
    Ok(())
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
    validate_anchor_with_sources(anchor, PACKAGED_MEMORY_ANCHOR_SOURCES)
}

fn validate_anchor_with_sources(
    anchor: &MemoryAnchor,
    sources: &[(&str, &str)],
) -> Result<(), String> {
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
    validate_currency_attestation("memory anchor", &anchor.id, &anchor.source)?;
    let Some((_, source_raw)) = sources.iter().find(|(path, _)| *path == anchor.source.path) else {
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
    if record.get("measurementKind").is_some() || record.get("ownedProcessMemory").is_some() {
        return Err("sampled accounting cannot satisfy a legacy residency/allocator anchor".into());
    }
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
    let record_reference_count = target
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
///
/// Parsed here so `cargo test` can hold the store to its declarations (every packaged anchor's
/// `(model, lane)` must be declared; an attestation must name the pin the file was derived at).
/// The runtime admission seam never reads it (sc-22738): there is no packaged accessor, and no
/// anchor, bound or curve lookup takes an `AnchorLoaderClosures`.
pub const PACKAGED_ANCHOR_LOADER_CLOSURES: &str =
    include_str!("../../../config/anchor-loader-closures.json");

/// Must equal the `digestVersion` of the checked-in file. Two versions answer different questions,
/// so a version bump is refused at parse rather than silently comparing digests derived under
/// different rules.
pub const ANCHOR_LOADER_CLOSURE_VERSION: &str = "anchor-loader-closure v2";

/// One `(model, backend lane)`'s declared loader closure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnchorLoaderClosure {
    /// For a CATALOG ALIAS only (sc-22724): the engine provider id this model id resolves to
    /// (`z_image_edit` → `z_image_turbo`). The inference tree never names the alias, so the
    /// derivation asks its literal rule of the engine id and hashes it into the closure text.
    /// Absent for every model that is its own engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_id: Option<String>,
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
    /// The declared loader-closure digest for one `(model, backend lane)`, or `None` when the
    /// coordinate is undeclared. A tooling/test accessor: the runtime compares no anchor against
    /// it (sc-22738).
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

/// A full 40-hex git revision, as the attestation names them.
fn is_revision(value: &str) -> bool {
    value.len() == 40 && value.chars().all(|c| c.is_ascii_hexdigit())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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
    /// The BINDING measured lower bound for one request, or `None` (sc-22738).
    ///
    /// Binding means: same identity on every axis a measured point is keyed on, and request
    /// geometry at or above the bound's on every axis. Among the survivors the LARGEST footprint
    /// wins — several bounds may cover one request once a cell has been stopped at more than one
    /// geometry, and the strongest true inequality is the useful one.
    ///
    /// Currency is NOT a conjunct, and the signature cannot express one: a bound whose loader
    /// closure has since moved refuses exactly what it refused when measured. The moved closure is
    /// the probe tooling's cue to stop at the cell again, never the runtime's cue to forget it.
    pub fn binding_exceeded_bound(&self, query: ExceededBoundQuery<'_>) -> Option<&ExceededBound> {
        self.exceeded_bounds
            .iter()
            .filter(|bound| {
                bound.model_id == query.model_id
                    && query
                        .model_family
                        .is_none_or(|family| bound.model_family == family)
                    && query.route.is_none_or(|route| bound.route == route)
                    && bound.provider == query.provider
                    && bound.backend == query.backend
                    && bound.tier == query.tier
                    && bound.transformer_variant == query.transformer_variant
                    && bound.decoder == query.decoder
                    && bound.mode == query.mode
                    && bound.overlay.as_deref() == query.overlay
                    && bound.reference_count == query.reference_count
                    && bound.covers_geometry(query.width, query.height, query.frames)
            })
            .max_by_key(|bound| bound.observed_footprint_bytes)
    }

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

    /// The unique VIDEO anchor for one `(model, backend lane, tier, transformer variant, decoder)`
    /// coordinate where the pipeline axes are OPTIONAL on both sides (sc-22736): an LTX-2.5
    /// request carries both and resolves exactly as [`Self::anchor_for`] does; a request on a
    /// video route that has no such axes at all (Wan 2.2, SCAIL-2, LTX-2.3) resolves the axis-free
    /// multi-frame anchor of its cell. The match is on `Option` EQUALITY, never on a wildcard: an
    /// axis-free anchor still answers no variant-keyed lookup, and an axis-keyed anchor answers no
    /// axis-free one. A single-frame axis-free row is an IMAGE anchor and is not a video anchor
    /// under any spelling — the mirror of [`Self::image_anchor_for`]'s refusal to hand a video
    /// anchor to an image request.
    pub fn video_anchor_for(
        &self,
        model_id: &str,
        backend: AnchorBackend,
        tier: &str,
        transformer_variant: Option<Ltx25TransformerVariant>,
        decoder: Option<Ltx25Decoder>,
    ) -> Option<&MemoryAnchor> {
        self.anchors.iter().find(|anchor| {
            anchor.model_id == model_id
                && anchor.backend == backend
                && anchor.tier == tier
                && anchor.transformer_variant == transformer_variant
                && anchor.decoder == decoder
                && anchor.geometry.frames > 1
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

/// Per-phase working-set estimates; admission uses their maximum. MLX image estimates return
/// active bytes and receive recapture uncertainty in the worker; MLX video estimates already
/// include it here. Candle retains its existing lane accounting.
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

/// MLX admission uses active allocations. Its cache is elastic, so a historical cache
/// high-water is not mandatory RAM. Preserve the other backend's existing accounting.
fn video_admission_bytes(bytes: i128, backend: AnchorBackend) -> Option<u64> {
    if backend == AnchorBackend::Mlx {
        widened_by(bytes, MLX_ACTIVE_RECAPTURE_SPREAD)
    } else {
        widened(bytes)
    }
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
            conditioning: video_admission_bytes(raw.conditioning.bytes, self.backend)?,
            denoise: video_admission_bytes(raw.denoise.bytes, self.backend)?,
            decode: video_admission_bytes(raw.decode.bytes, self.backend)?,
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
        // Every per-token coefficient in this law is an LTX-2.5 pipeline fact keyed on the
        // transformer variant and decoder. An anchor that states no such axes has no fitted law
        // (sc-22736): a video route without those axes (Wan 2.2, SCAIL-2, LTX-2.3) is priced ONLY at
        // and below its own measured point, and an image anchor (single frame, sc-22509) is not a
        // video anchor at all. Consulted BEFORE the underived flag below, because the ONLY reason
        // a video anchor is stamped underived is this very absence of axes — and the measured-point
        // bound is exactly the "validates its measured point and prices nothing beyond it" the
        // reason states. It fits nothing, so, like the image law, it does not consult the flag.
        if self.transformer_variant.is_none() || self.decoder.is_none() {
            return self.measured_point_bound_raw(request);
        }
        // An anchor marked underived validates its measured point and prices nothing (epic 22505
        // feature-end fix round): every lane law carries this refusal so the store's statement and
        // the runtime behaviour cannot disagree.
        if self.underived_reason.is_some() {
            return None;
        }
        if !self.measured_regime_prices(request) {
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

    /// REGIME GUARD (anchor vs request), shared by the fitted video law and the measured-point
    /// bound: a phase priced from the anchor's own measured intercept requires the anchor to have
    /// run that phase UNBOUNDED. A deferred-materialization anchor carries no transformer
    /// residency in its conditioning peak; a windowed anchor carries one block instead of the
    /// whole transformer in its denoise peak; a tiled anchor carries a tile workspace instead of
    /// the full decode working set. Reusing any of those for an unbounded request would
    /// under-estimate by the whole omitted residency.
    fn measured_regime_prices(&self, request: AnchorDeriveRequest) -> bool {
        let anchor_deferred = self.load_shape == AnchorLoadShape::DeferredMaterialization;
        !((!request.deferred_materialization && anchor_deferred)
            || (!request.transformer_windowed && self.measured_regime.transformer_windowed)
            || (!request.decode_tiled && self.measured_regime.decode_tiled))
    }

    /// The MEASURED-POINT BOUND for a video anchor that states no pipeline axes (sc-22736): Wan 2.2,
    /// SCAIL-2 and LTX-2.3 anchors, whose routes have no `(transformer variant, decoder)` and for
    /// which no per-token coefficient has been measured.
    ///
    /// Such an anchor prices nothing BEYOND its measured point — the law above has no slope for it
    /// and none is invented here — but it does bound every request the measured render DOMINATES:
    /// a request at no more output pixels and no more frames than the anchor, under a regime the
    /// anchor's measured intercepts price ([`Self::measured_regime_prices`]), cannot exceed the
    /// anchor's own per-phase peaks, so each phase is estimated AT the anchor's measured intercept
    /// (widened by the shared allocator margin by the caller, like every anchored term). Pixels
    /// rather than latent tokens because the token lattice is a per-model fact this law does not
    /// know, and pixels are lattice-free and aspect-symmetric — a portrait bucket is bounded by
    /// its landscape twin; frames per axis because the temporal lattice is likewise unknown.
    ///
    /// A request bounding a phase the anchor measured unbounded (a tiled decode against an untiled
    /// anchor, a deferred request against an eager one) is still priced at the unbounded measured
    /// intercept: bounding only shrinks that phase's working set, so the measured peak remains an
    /// upper bound — no architecture constant is substituted, because every such constant in this
    /// module is an LTX-2.5 fact.
    ///
    /// `None` for a single-frame anchor (an image anchor is not a video anchor), a degenerate
    /// geometry, a request the anchor does not dominate, or a regime it cannot price — each of
    /// which leaves the caller on its analytic floor.
    fn measured_point_bound_raw(
        &self,
        request: AnchorDeriveRequest,
    ) -> Option<RawVideoPhaseEstimates> {
        if self.geometry.frames <= 1
            || request.width == 0
            || request.height == 0
            || request.frames == 0
        {
            return None;
        }
        let request_pixels = u64::from(request.width) * u64::from(request.height);
        let anchor_pixels = u64::from(self.geometry.width) * u64::from(self.geometry.height);
        if request_pixels > anchor_pixels || request.frames > self.geometry.frames {
            return None;
        }
        if !self.measured_regime_prices(request) {
            return None;
        }
        let phase = |bytes: u64| RawPhaseEstimate {
            bytes: i128::from(bytes),
            anchored: true,
        };
        Some(RawVideoPhaseEstimates {
            conditioning: phase(self.phase_active_peak_bytes.conditioning),
            denoise: phase(self.phase_active_peak_bytes.denoise),
            decode: phase(self.phase_active_peak_bytes.decode),
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
        let windowed_transformer = i128::from(windowed_transformer_bytes(
            components.transformer,
            regime.transformer_window,
            facts.transformer_blocks,
        ));
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

    /// The sc-22509 shallow entry point onto [`Self::derive_phase_peaks`], kept for the fixtures
    /// written against it. Since sc-22667 NO worker lane prices through it: the shared candle
    /// ladder (sc-22664) and the Krea 2 Turbo lane both call the law directly with the rung's own
    /// regime (`RequestRegime` from the selected parameters) and the contract's architecture
    /// facts, so a rung's tile, chunk and window reach the estimate.
    ///
    /// Pins the lane (a non-candle anchor is refused) and keeps the shallow-anchor asymmetry the
    /// candle consumers were written against: the anchor must be the staged composition with no
    /// deeper rung engaged, and the graded candidate must engage `staged_residency`. With no rung
    /// parameters and no facts the law prices every composition containing the anchor's at the
    /// shallow staged working set, which upper-bounds the deeper rungs. An anchor carrying
    /// [`MemoryAnchor::underived_reason`] is refused, as before.
    ///
    /// sc-22734 adds the one cell shape that asymmetry does not describe: a provider whose contract
    /// classifies `staged_residency` STRUCTURALLY NOT APPLICABLE has no staged composition at all,
    /// so its anchor is resident and prices a resident candidate. See
    /// [`MemoryAnchor::staged_residency_structurally_not_applicable`] for why that is sound rather
    /// than the refused resident case. The two shapes are mutually exclusive and a row that mixes
    /// them (staged anchor on an exempt cell, resident anchor on a non-exempt one) is still
    /// refused, so no packaged row's disposition moves.
    pub fn derive_image_phase_peaks(
        &self,
        request: AnchorImageDeriveRequest,
        components: ComponentBytes,
    ) -> Option<AnchorDerivedPhases> {
        if self.underived_reason.is_some() || self.backend != AnchorBackend::Candle {
            return None;
        }
        if self.measured_regime.decode_tiled
            || self.measured_regime.attention_chunked
            || self.measured_regime.transformer_windowed
        {
            return None;
        }
        // Two admissible shapes, and a cell is only ever one of them
        // ([`MemoryAnchor::staged_residency_structurally_not_applicable`]).
        let regime = match (
            self.measured_regime.staged,
            self.staged_residency_structurally_not_applicable,
        ) {
            // The ordinary cell: a staged anchor prices a staged candidate.
            (true, false) => {
                if !request.staged_residency {
                    return None;
                }
                RequestRegime::staged()
            }
            // sc-22734: a cell whose engine has NO staged composition. Its anchor is the resident
            // render — the only one that exists — and it prices a resident candidate. There is no
            // co-resident encoder above it to go unpriced, which is the whole reason the staged
            // asymmetry exists, so the law prices it the way the MLX lane prices its residents.
            (false, true) => {
                if request.staged_residency {
                    return None;
                }
                RequestRegime::resident()
            }
            // A staged anchor on a cell that declares staging impossible, or a resident anchor on a
            // cell that has a staged composition it should have been captured in: both are records
            // the law cannot honestly price, so both stay refused.
            _ => return None,
        };
        self.derive_phase_peaks(
            &ImageDeriveRequest::new(request.width, request.height, regime),
            components,
            ArchitectureFacts::default(),
        )
    }

    /// Recover each usable resident MLX phase independently. A cached conditioning capture may
    /// omit its encoder weights; that phase remains unknown without discarding a valid denoise
    /// or decode observation. These are activation residues, not complete optimized measurements.
    pub fn derive_mlx_image_activation_residues(
        &self,
        request: AnchorMlxImageDeriveRequest,
        components: ComponentBytes,
    ) -> [Option<u64>; 3] {
        if self.underived_reason.is_some()
            || self.backend != AnchorBackend::Mlx
            || self.load_shape != AnchorLoadShape::EagerMaterialization
            || self.measured_regime.staged
            || self.measured_regime.decode_tiled
            || self.measured_regime.attention_chunked
            || self.measured_regime.transformer_windowed
            || self.transformer_variant.is_some()
            || self.decoder.is_some()
            || self.geometry.frames != 1
            || self.geometry.width == 0
            || self.geometry.height == 0
            || request.width == 0
            || request.height == 0
        {
            return [None; 3];
        }
        let resident = components
            .conditioning
            .saturating_add(components.transformer)
            .saturating_add(components.decoder);
        let pixels = u128::from(request.width) * u128::from(request.height);
        let anchor_pixels = u128::from(self.geometry.width) * u128::from(self.geometry.height);
        let scale = |residue: u64, quadratic: bool| {
            let ratio = |numerator: u128, denominator: u128| {
                u128::from(residue)
                    .checked_mul(numerator)
                    .map(|n| n.div_ceil(denominator))
            };
            let linear = ratio(pixels, anchor_pixels)?;
            let scaled = if quadratic {
                linear.max(ratio(pixels * pixels, anchor_pixels * anchor_pixels)?)
            } else {
                linear
            };
            // Retain observations below their capture geometry; the fixed/workspace split is unknown.
            u64::try_from(scaled.max(u128::from(residue))).ok()
        };
        [
            self.phase_active_peak_bytes
                .conditioning
                .checked_sub(resident),
            self.phase_active_peak_bytes
                .denoise
                .checked_sub(resident)
                .and_then(|r| scale(r, true)),
            self.phase_active_peak_bytes
                .decode
                .checked_sub(resident)
                .and_then(|r| scale(r, false)),
        ]
    }

    /// Derive the MLX image lane's non-reclaimable active working set.
    ///
    /// Reclaimable cache is retained as capture provenance only. Adding it to a physical-memory
    /// requirement contradicts `memory_calibration::Phase::non_reclaimable_bytes`: cache shrinks
    /// under pressure and its boundary reading is not simultaneous with the active high-water mark.
    /// This resident entry point preserves the source regime and materialization guards.
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
        self.derive_phase_peaks(
            &ImageDeriveRequest::new(request.width, request.height, RequestRegime::resident()),
            components,
            ArchitectureFacts::default(),
        )
    }
}

/// The transformer bytes a `bounded_transformer_residency` rung keeps RESIDENT: the law's window
/// share `transformer x min(window, blocks) / blocks`, rounded up (sc-22663, epic 22657 E3; the
/// worker's MLX estimate floor states the same share since sc-22667 whenever the facts carry a
/// block count, so both image lanes price one windowed residency). With no window the whole
/// transformer is resident; with a window but no block count — the default facts at this pin —
/// the share is UNKNOWABLE and the LAW keeps the whole transformer, the erring-large reading,
/// never zero. (The MLX floor's no-block-count arm is a separate, worker-side decision: it keeps
/// its pre-epic accounting so shipped admissions do not move before the pin bump.)
pub fn windowed_transformer_bytes(
    transformer: u64,
    window: Option<u32>,
    blocks: Option<u32>,
) -> u64 {
    match (window, blocks) {
        (Some(window), Some(blocks)) if blocks > 0 && window > 0 => u64::try_from(scale_up(
            i128::from(transformer),
            i128::from(window.min(blocks)),
            i128::from(blocks),
        ))
        .unwrap_or(transformer),
        _ => transformer,
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

// ---------------------------------------------------------------------------------------------
// MLX admission policy shared by the worker's fit gate and the memory adapter (sc-22738)
// ---------------------------------------------------------------------------------------------
//
// The memory adapter's ordinary LTX-2.3 capture admits through the SAME projection the worker's
// gate would build for the cell (epic 22723 sc-22738, review round 2). The three inputs that
// projection needs beyond the provider contract — the fixed unified-memory reserve, the generic
// activation allowance and the floor's weights arithmetic — were private to
// `crates/sceneworks-worker/src/{fit_gate,mlx_fit_gate}.rs`, and the adapter's first cut restated
// two of them as fresh literals and the third as a hand transcription. They are declared ONCE here;
// the worker reads its constants from these and both crates call [`floor_weights_bytes`]. A
// change to any of them therefore moves the worker's admission and the adapter's projection
// together, or not at all.

/// Bytes per binary gigabyte (GiB) — `hw.memsize / 1024³`, the unit every MLX budget figure is
/// stated in (`gpu::total_unified_memory_gb`, the epic's measured on-disk tables).
pub const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// The MLX lane's fixed unified-memory reserve: `mlx_fit_gate::live_request_budget` presents it as
/// `MemoryBudget::reserved_headroom_bytes` for every non-Mage engine, and
/// `MemoryBudget::effective_bytes` subtracts it before the shared predicate compares. Exact
/// verified cells do not use this number for their foreign demand (that is derived from captured
/// MLX/wired limits); it is Decision 2's promise that no-record/out-of-envelope/stale requests
/// take the legacy reserve rather than silently adopting a new policy.
pub const LEGACY_UNIFIED_FALLBACK_RESERVE_GB: f64 = 2.0;

/// Dedicated-VRAM allocator/context slack the Candle lane charges once on top of a per-tier
/// measured peak (`fit_gate::dedicated_vram_reserve`, `vram_gate::HEADROOM_GB`). Declared here so
/// the catalog's memory guidance (`crate::memory_guidance`) quotes the same reserve the worker
/// admits with; the worker reads this constant rather than declaring a second copy.
pub const DEDICATED_VRAM_ALLOCATOR_SLACK_GB: f64 = 2.0;

/// The Candle lane's same-cell recapture allowance (`ladder_margin_policy::CANDLE_RECAPTURE_SPREAD`
/// documents its derivation and pins it against the evidence corpus). Declared here so the catalog's
/// SDXL endpoint guidance applies the allowance the worker admits with.
pub const CANDLE_ACTIVE_RECAPTURE_SPREAD: f64 = 0.02;

/// The MLX lane's generic flat activation allowance (`HEADROOM_GB`, sc-10863): the max
/// common-case measured 1024² transient (14.04 GiB) plus a ~4 GiB OS/app reserve. The video
/// lane's estimate floor is `floor weights + this` (`video_admission::floor_phase_peaks`), and a
/// provider decode profile only ever RAISES that floor (`profiled_floor_phase_peaks`).
pub const MLX_GENERIC_HEADROOM_GB: f64 = 18.0;

/// GiB to bytes, rounded to the nearest byte and clamped into `u64` — the worker's own conversion
/// (`mlx_fit_gate::gib_to_bytes`), so an allowance stated in GiB means the same byte count on both
/// sides of the seam.
pub fn gib_to_bytes(gib: f64) -> u64 {
    (gib * BYTES_PER_GIB).round().clamp(0.0, u64::MAX as f64) as u64
}

/// The contract facts the floor's weights term reads, lifted off `gen_core::MemoryProviderContract`
/// by each caller (this crate does not see gen-core's types). Every field is a declaration the
/// provider makes about itself; nothing here is a tuned coefficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FloorWeightsFacts {
    /// `asset_facts.conditioning_bytes` — the conditioning encoder stack.
    pub conditioning_bytes: u64,
    /// `asset_facts.base_bytes` — every resident weight of a plain render.
    pub base_bytes: u64,
    /// `asset_facts.transformer_bytes` — the load-exact transformer, evictable sub-stack included.
    pub transformer_bytes: u64,
    /// Bytes the provider declares it drops from INSIDE `transformer_bytes` before the declaring
    /// phase reaches steady state: `transformer_bytes - steady_state_transformer_bytes()`. Zero for
    /// every provider that has not adopted the sub-stack vocabulary.
    pub intra_transformer_evicted_bytes: u64,
    /// The sum of every auxiliary resident component (control branches, adapter stacks, …) the
    /// composition does NOT bound: `resident_components()` filtered to `kind.is_auxiliary()` and
    /// to `bounded_by` either absent or naming a rung the composition leaves disengaged.
    pub auxiliary_resident_bytes: u64,
}

/// Which reductions the engaged composition applies to [`FloorWeightsFacts`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FloorWeightsComposition {
    /// `StagedResidency` engaged.
    pub staged: bool,
    /// `BoundedTransformerResidency` engaged.
    pub bounded_transformer: bool,
    /// What rung 4 keeps resident of the transformer once it has removed it: 0 on the video lane,
    /// the law's window share on the image lane. Ignored unless `bounded_transformer`.
    pub resident_window_bytes: u64,
}

/// The floor's per-rung WEIGHTS term (sc-18096), derived only from the provider's own
/// declarations:
///
/// * `staged` ⇒ the co-residency drop the rung exists for: the resident working set is the larger
///   of the conditioning stack and everything else (the `staged_weights_gb` split the load-time
///   gate has always used).
/// * `bounded_transformer` ⇒ the transformer's declared bytes leave the resident floor and the
///   rung's RESIDENT WINDOW comes back in, clamped at the load-exact transformer so no lane can
///   state more than it loads.
/// * The two reductions are EXCLUSIVE, not sequential. The staged reduction's clamp holds the lump
///   at `transformer_bytes` precisely so the precompute instant stays covered — and rung 4 then
///   removes that same `transformer_bytes`, sub-stack included. Applying both would leave
///   `max(0, decoder − evicted)` where the answer is `decoder`.
/// * A declared intra-transformer eviction leaves the floor only on the STAGED branch, and only
///   down to the load-exact transformer (sc-19721): the drop lowers the steady state, not the
///   peak, so it may only cancel against bytes provably NOT co-resident with the precompute instant.
/// * Rungs 2 and 3 bound TRANSIENTS, not weights, and take no reduction here.
/// * Unbounded auxiliary components stay resident.
pub fn floor_weights_bytes(facts: FloorWeightsFacts, composition: FloorWeightsComposition) -> u64 {
    let conditioning = facts.conditioning_bytes;
    // The load-exact non-conditioning working set: what the transformer's own phase holds while the
    // evictable sub-stack is still materialized.
    let heavy_load_exact = facts.base_bytes.saturating_sub(conditioning);
    let heavy = if composition.staged && !composition.bounded_transformer {
        heavy_load_exact
            .saturating_sub(facts.intra_transformer_evicted_bytes)
            .max(facts.transformer_bytes)
    } else if composition.bounded_transformer {
        heavy_load_exact
            .saturating_sub(facts.transformer_bytes)
            .saturating_add(
                composition
                    .resident_window_bytes
                    .min(facts.transformer_bytes),
            )
    } else {
        heavy_load_exact
    };
    let base = if composition.staged {
        conditioning.max(heavy)
    } else {
        conditioning.saturating_add(heavy)
    };
    base.saturating_add(facts.auxiliary_resident_bytes)
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
    ///
    /// The pipeline axes are OPTIONAL (sc-22736): a request on a route without them resolves its
    /// axis-free anchor through [`Self::video_anchor_for`] and is priced by that anchor's
    /// measured-point bound; the sibling+delta fall-through below is LTX-2.5's — it crosses
    /// `(variant, decoder)` axes — so an axis-free request that has no exact anchor takes no
    /// sibling and keeps its floor.
    pub fn derive_video_phase_peaks_for_cell(
        &self,
        model_id: &str,
        backend: AnchorBackend,
        tier: &str,
        transformer_variant: Option<Ltx25TransformerVariant>,
        decoder: Option<Ltx25Decoder>,
        request: AnchorDeriveRequest,
    ) -> Option<AnchorCellDerivation<'_>> {
        if let Some(anchor) =
            self.video_anchor_for(model_id, backend, tier, transformer_variant, decoder)
        {
            return Some(AnchorCellDerivation {
                phases: anchor.derive_video_phase_peaks(request)?,
                anchor,
                delta_bytes: 0,
            });
        }
        let (Some(transformer_variant), Some(decoder)) = (transformer_variant, decoder) else {
            return None;
        };
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
                    conditioning: video_admission_bytes(priced(raw.conditioning), backend)?,
                    denoise: video_admission_bytes(priced(raw.denoise), backend)?,
                    decode: video_admission_bytes(priced(raw.decode), backend)?,
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
#[path = "memory_anchor/tests.rs"]
mod tests;
