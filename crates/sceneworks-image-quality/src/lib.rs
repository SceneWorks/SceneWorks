//! Tier-0 pixel extraction (epic 6529 "Dataset Doctor", sc-6532).
//!
//! The decode-bearing half of Tier-0: turn an image into the per-image scalars that the *pure*
//! decision logic in [`sceneworks_core::dataset_quality`] consumes. It lives in its own crate —
//! not the decode-free core, not the GPU-laden worker — so it builds and tests anywhere on just
//! the pure-Rust `image` (png/jpeg/webp) + `imageproc` + `image_hasher` stack, with no Metal/CUDA.
//! Both the worker and the API's synchronous readiness path can depend on it.
//!
//! Blur + exposure are measured on the **center-crop→bucket-resize the trainer actually feeds**
//! (mirrors candle-gen's `train/dataset.rs` kernel): that makes the variance comparable across
//! source resolutions and folds in upscale-to-mush (a tiny image blown up to the bucket reads as
//! soft, which is the correct signal). The perceptual hash is taken on the full image — it
//! self-normalizes by downscaling internally. See `docs/sc-6530/dataset-doctor-metrics.md`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use image::{imageops::FilterType, DynamicImage, GrayImage, Luma, RgbImage};
use image_hasher::{HashAlg, Hasher, HasherConfig};
use sceneworks_core::dataset_quality::{
    build_readiness_report, evaluate_tier0, plan_duplicate_removal, AestheticEvaluation,
    AestheticPredictor, CaptionAlignmentEvaluation, DatasetReadinessReport, DuplicateCandidate,
    DuplicateRemovalPlan, IdentityEvaluation, ItemQualityInput, MetricDistribution, QualityCheck,
    QualityFlag, ReadinessDistributions, Severity, Tier0Scalars, Tier0Thresholds, Tier1Evaluation,
};

/// One-tap pixel-rewriting fixes (sc-6539 smart-crop + EXIF-strip).
pub mod transform;

/// Luma at or below this counts as crushed-to-black (8-bit).
const SHADOW_CUTOFF: u8 = 4;
/// Luma at or above this counts as blown-to-white (8-bit).
const HIGHLIGHT_CUTOFF: u8 = 251;

/// Focus is measured per-tile over an `FOCUS_TILE_GRID`×`FOCUS_TILE_GRID` grid, then reduced to a
/// single "is *any* region sharp?" number — so a sharp subject on a flat/dark background is not
/// dragged soft by the background's near-zero response (sc-8563). 8×8 = 64 tiles: a subject
/// occupying even a modest fraction of the frame still lands in several tiles.
const FOCUS_TILE_GRID: u32 = 8;
/// Reduce the per-tile variances to their `FOCUS_TILE_PERCENTILE` quantile — the sharp-region
/// signal, but robust to a single high-frequency outlier tile (a strict max would let one noisy
/// JPEG block or specular edge rescue an otherwise-blurry frame, sc-8563).
const FOCUS_TILE_PERCENTILE: f64 = 0.90;

/// Luma band counted as "well exposed" — neither crushed toward black nor blown toward white. A
/// low-key portrait keeps its lit subject inside this band even while the background clips, which is
/// what separates an intentional dark backdrop from a genuinely under/over-exposed frame (sc-8563).
const WELL_EXPOSED_LOW: u8 = 24;
const WELL_EXPOSED_HIGH: u8 = 231;

/// The one pinned perceptual-hash configuration. A fixed algorithm + size means every stored hash
/// is the same byte length and Hamming distances are comparable across the whole dataset — the
/// invariant the near-duplicate clustering in `sceneworks_core::dataset_quality` relies on.
fn dataset_hasher() -> Hasher {
    HasherConfig::new()
        .hash_alg(HashAlg::Gradient) // dHash: cheap, robust to small edits, good for near-dup
        .hash_size(8, 8)
        .to_hasher()
}

/// The bundled LAION-Aesthetics V2 MLP head (sc-6537), parsed once. Extracted (head only) from
/// `shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE` (Apache-2.0; see
/// `assets/README.md`) — it scores the L2-normalized CLIP ViT-L/14 image embedding produced by the
/// dataset-analysis job. Lives here (the GPU-free host-eval crate) so the readiness path can score
/// without a model server.
pub fn aesthetic_predictor() -> &'static AestheticPredictor {
    static PREDICTOR: std::sync::OnceLock<AestheticPredictor> = std::sync::OnceLock::new();
    PREDICTOR.get_or_init(|| {
        const BYTES: &[u8] =
            include_bytes!("../assets/aesthetic-v2-sac-logos-ava1-l14.safetensors");
        AestheticPredictor::from_safetensors_bytes(BYTES)
            .expect("the bundled aesthetic predictor asset parses")
    })
}

/// Decode an image, transcoding a valid-but-unsupported format (AVIF/HEIC/HEIF/TIFF/BMP/GIF) to a
/// temp PNG first (sc-6143). `decode` runs directly on `path` on the fast path; only when that fails
/// *and* the bytes sniff as a recognized format the pure-Rust `image` build can't decode do we shell
/// out to the shared [`sceneworks_core::media_convert`] transcoder and re-run `decode` on the PNG.
///
/// Import normalizes new uploads to PNG, but assets that predate that change — or reach this crate by
/// a path that skips normalization — would otherwise fail here (`The image format Avif is not
/// supported`). This mirrors the worker's `decode_image_any` backstop so the API's synchronous
/// Dataset Doctor paths degrade to "transcode + run" instead of erroring.
pub(crate) fn decode_transcoding<F>(path: &Path, decode: F) -> image::ImageResult<DynamicImage>
where
    F: Fn(&Path) -> image::ImageResult<DynamicImage>,
{
    use sceneworks_core::media_convert::{sniff_image_kind_at, transcode_to_png};

    match decode(path) {
        Ok(image) => Ok(image),
        Err(decode_error) => {
            // Only transcode a format we recognize as valid-but-unsupported; a natively-supported
            // format (or unrecognized bytes) means a genuine decode failure — surface it unchanged.
            match sniff_image_kind_at(path) {
                Some(kind) if !kind.is_natively_supported() => {
                    let dir = tempfile::tempdir().map_err(image::ImageError::IoError)?;
                    let png = dir.path().join("decoded.png");
                    if let Err(error) = transcode_to_png(path, &png) {
                        return Err(image::ImageError::IoError(std::io::Error::other(
                            error.to_string(),
                        )));
                    }
                    decode(&png)
                }
                _ => Err(decode_error),
            }
        }
    }
}

/// Decode the image at `path` and extract its Tier-0 scalars. `bucket_edge` is the trainer's
/// target square resolution (the size blur + exposure are measured at).
pub fn extract_tier0_scalars(path: &Path, bucket_edge: u32) -> image::ImageResult<Tier0Scalars> {
    Ok(scalars_from_image(
        &decode_transcoding(path, |p: &Path| image::open(p))?,
        bucket_edge,
    ))
}

/// Extract Tier-0 scalars from an already-decoded image — the testable core, no IO.
pub fn scalars_from_image(image: &DynamicImage, bucket_edge: u32) -> Tier0Scalars {
    let phash = dataset_hasher().hash_image(image).as_bytes().to_vec();

    // Measure sharpness + exposure on exactly what the trainer feeds in. (If the metrics below
    // change meaning, bump `sceneworks_core::dataset_quality::TIER0_METRICS_VERSION` so persisted
    // caches recompute instead of being judged under the old definition.)
    let trainer_gray = trainer_grayscale(image, bucket_edge.max(1));
    let (shadow_clip, highlight_clip) = exposure_clip(&trainer_gray);
    Tier0Scalars {
        blur_variance: tile_peak_focus(&trainer_gray),
        shadow_clip,
        highlight_clip,
        well_exposed_fraction: well_exposed_fraction(&trainer_gray),
        phash,
    }
}

/// One item's inputs for [`compute_readiness`]. `cached_scalars` is the validated cache the caller
/// chose to reuse (via `CachedTier0Scalars::valid_for`); when it is `None`, the image at
/// `image_path` is decoded.
pub struct ReadinessItem {
    pub item_id: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub content_hash: Option<String>,
    pub image_path: Option<PathBuf>,
    pub cached_scalars: Option<Tier0Scalars>,
    /// Checks the user dismissed for this image (sc-6534), already resolved by the caller against the
    /// current content hash (`QualityAck::effective_checks`). Threaded into the evaluation so a
    /// dismissed finding drops out of the rollups.
    pub acknowledged: Vec<QualityCheck>,
}

/// Whether a stored image path can still carry embedded metadata (sc-6539). The strip-metadata fix
/// re-encodes to a metadata-free PNG, so anything already stored as PNG is treated as clean; every
/// other accepted raster format (JPEG/HEIC/WebP/TIFF/…) can carry EXIF/GPS/ICC and is offered the
/// one-tap strip. Path-string only (no decode), so it is free even for cached-scalar items.
fn path_can_carry_metadata(path: &Path) -> bool {
    !path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
}

/// Compute the dataset readiness report (sc-6533), decoding Tier-0 scalars for any item without a
/// reusable cached value. Returns the report plus the freshly-extracted `(item_id, scalars)` pairs
/// so the caller can persist them as the content-hash + bucket-keyed cache. An item whose image
/// fails to decode gets a `Decode` warning, so it can never pass as "technically fine".
// Each optional eval (`tier1`/`alignment`/`aesthetic`/`identity`) is a distinct, independently-present
// fold input the caller assembles from separate sidecars — a struct would only relocate the arity.
#[allow(clippy::too_many_arguments)]
pub fn compute_readiness(
    items: &[ReadinessItem],
    bucket_edge: u32,
    min_items: u32,
    thresholds: &Tier0Thresholds,
    tier1: Option<&Tier1Evaluation>,
    alignment: Option<&CaptionAlignmentEvaluation>,
    aesthetic: Option<&AestheticEvaluation>,
    identity: Option<&IdentityEvaluation>,
) -> (DatasetReadinessReport, Vec<(String, Tier0Scalars)>) {
    let mut inputs = Vec::with_capacity(items.len());
    let mut extracted = Vec::new();
    let mut decode_failed = Vec::new();

    for item in items {
        let scalars = if let Some(cached) = &item.cached_scalars {
            Some(cached.clone())
        } else if let Some(path) = &item.image_path {
            match extract_tier0_scalars(path, bucket_edge) {
                Ok(scalars) => {
                    extracted.push((item.item_id.clone(), scalars.clone()));
                    Some(scalars)
                }
                Err(_) => {
                    decode_failed.push(item.item_id.clone());
                    None
                }
            }
        } else {
            None
        };
        inputs.push(ItemQualityInput {
            item_id: item.item_id.clone(),
            width: item.width,
            height: item.height,
            content_hash: item.content_hash.clone(),
            scalars,
            acknowledged: item.acknowledged.clone(),
        });
    }

    let mut evaluation = evaluate_tier0(&inputs, bucket_edge, min_items, thresholds);
    for entry in &mut evaluation.items {
        if decode_failed.contains(&entry.item_id) {
            // Injected after the ack post-pass, so a decode failure is never acknowledgeable here —
            // and the API also strips `Decode` from stored acks. An undecodable image stays a real
            // warning that drags the technical share down (it can't pass as "technically fine").
            entry.flags.push(QualityFlag {
                check: QualityCheck::Decode,
                severity: Severity::Warn,
                value: None,
                threshold: None,
                peers: Vec::new(),
                acknowledged: false,
            });
        }
    }

    // identity (sc-6538): the API layer holds the face sidecar, builds the `IdentityEvaluation` for a
    // Person dataset, and threads it here; `None` (non-person / no sidecar) leaves the face sub-score
    // and flags untouched.
    let mut report = build_readiness_report(evaluation, tier1, alignment, aesthetic, identity);
    report.distributions = build_distributions(&inputs, thresholds);
    report.duplicate_removal = plan_dataset_duplicate_removal(&report, &inputs);
    // Strip-metadata gating (sc-6539): the one-tap fix re-encodes to a metadata-free PNG, so the
    // Doctor should only offer it where it would actually change something. An item already stored as
    // PNG is clean; every other accepted raster format (JPEG/HEIC/WebP/TIFF/…) can still carry
    // EXIF/GPS/ICC. Marking only those makes the action stop re-appearing once an item is stripped.
    let strippable: std::collections::HashSet<&str> = items
        .iter()
        .filter(|item| {
            item.image_path
                .as_deref()
                .is_some_and(path_can_carry_metadata)
        })
        .map(|item| item.item_id.as_str())
        .collect();
    for entry in &mut report.items {
        entry.metadata_strippable = strippable.contains(entry.item_id.as_str());
    }
    (report, extracted)
}

/// One-tap "drop duplicates" plan (sc-6539), built from the report's exact/near-duplicate flags and
/// the sharpness/size already gathered for evaluation, so the web carries it in the one report
/// payload. Only `ExactDuplicate` + `NearDuplicate` (pHash) feed the auto-plan — CLIP
/// `NearDuplicateEmbedding` is excluded on purpose (same-content/different-crop frames are legitimate
/// training variety, and removing them would fight the sibling `LowDiversity` check). Acknowledged
/// duplicates are skipped, so an image the user chose to keep never lands in the plan. `None` when
/// nothing is safely removable.
fn plan_dataset_duplicate_removal(
    report: &DatasetReadinessReport,
    inputs: &[ItemQualityInput],
) -> Option<DuplicateRemovalPlan> {
    // Sharpness (Laplacian variance) + pixel count rank which copy to keep; both already on hand.
    let signal: HashMap<&str, (Option<f64>, Option<u64>)> = inputs
        .iter()
        .map(|input| {
            let pixels = input
                .width
                .zip(input.height)
                .map(|(w, h)| u64::from(w) * u64::from(h));
            (
                input.item_id.as_str(),
                (input.scalars.as_ref().map(|s| s.blur_variance), pixels),
            )
        })
        .collect();

    let candidates: Vec<DuplicateCandidate> = report
        .items
        .iter()
        .filter_map(|item| {
            let peers: Vec<String> = item
                .flags
                .iter()
                .filter(|flag| {
                    !flag.acknowledged
                        && matches!(
                            flag.check,
                            QualityCheck::ExactDuplicate | QualityCheck::NearDuplicate
                        )
                })
                .flat_map(|flag| flag.peers.iter().cloned())
                .collect();
            if peers.is_empty() {
                return None;
            }
            let (sharpness, pixels) = signal
                .get(item.item_id.as_str())
                .copied()
                .unwrap_or((None, None));
            Some(DuplicateCandidate {
                item_id: item.item_id.clone(),
                duplicate_peers: peers,
                sharpness,
                pixels,
            })
        })
        .collect();

    plan_duplicate_removal(&candidates)
}

/// Per-metric distributions for the Advanced view (sc-6534), built from the scalars already gathered
/// for evaluation — so the report carries them and the web needs no second fetch. `None` when no
/// item has scalars (empty or wholly-undecodable set).
fn build_distributions(
    inputs: &[ItemQualityInput],
    thresholds: &Tier0Thresholds,
) -> Option<ReadinessDistributions> {
    let scalars: Vec<&Tier0Scalars> = inputs.iter().filter_map(|i| i.scalars.as_ref()).collect();
    if scalars.is_empty() {
        return None;
    }
    Some(ReadinessDistributions {
        // Sharpness: higher is better, judged against the absolute blur floor.
        blur_variance: MetricDistribution {
            values: scalars.iter().map(|s| s.blur_variance).collect(),
            threshold: Some(thresholds.blur_floor),
            higher_is_better: true,
        },
        // Clip fractions: lower is better, judged against the exposure-clip ceiling.
        shadow_clip: MetricDistribution {
            values: scalars.iter().map(|s| s.shadow_clip).collect(),
            threshold: Some(thresholds.exposure_clip_fraction),
            higher_is_better: false,
        },
        highlight_clip: MetricDistribution {
            values: scalars.iter().map(|s| s.highlight_clip).collect(),
            threshold: Some(thresholds.exposure_clip_fraction),
            higher_is_better: false,
        },
    })
}

/// Center-crop to a square and resize to `edge`×`edge` (Lanczos3), then grayscale — mirroring the
/// trainer's `load_image_tensor` kernel so the scalars describe what training sees.
fn trainer_grayscale(image: &DynamicImage, edge: u32) -> GrayImage {
    let rgb: RgbImage = image.to_rgb8();
    let side = rgb.width().min(rgb.height()).max(1);
    let x0 = (rgb.width() - side) / 2;
    let y0 = (rgb.height() - side) / 2;
    let cropped = image::imageops::crop_imm(&rgb, x0, y0, side, side).to_image();
    let resized = image::imageops::resize(&cropped, edge, edge, FilterType::Lanczos3);
    DynamicImage::ImageRgb8(resized).into_luma8()
}

/// The 4-connected Laplacian response over the whole image — the classic focus operator. Computed
/// once and then tiled by [`tile_peak_focus`]; high-frequency detail responds large, a flat or
/// out-of-focus region responds near zero.
fn laplacian_response(gray: &GrayImage) -> image::ImageBuffer<Luma<i32>, Vec<i32>> {
    let kernel: [i32; 9] = [0, 1, 0, 1, -4, 1, 0, 1, 0];
    imageproc::filter::filter3x3(gray, &kernel)
}

/// Variance of the Laplacian-response values inside `[x0, x0+w) × [y0, y0+h)`. Empty rect ⇒ 0.
fn response_tile_variance(
    response: &image::ImageBuffer<Luma<i32>, Vec<i32>>,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
) -> f64 {
    let mut count = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for y in y0..(y0 + h) {
        for x in x0..(x0 + w) {
            let value = f64::from(response.get_pixel(x, y)[0]);
            sum += value;
            sum_sq += value * value;
            count += 1.0;
        }
    }
    if count == 0.0 {
        return 0.0;
    }
    let mean = sum / count;
    (sum_sq / count - mean * mean).max(0.0)
}

/// Tile-based peak focus (sc-8563): split the image into `FOCUS_TILE_GRID`² tiles, take each tile's
/// Laplacian variance, and reduce to the `FOCUS_TILE_PERCENTILE` quantile — "is the sharpest region
/// sharp?". A sharp subject on a flat/dark background scores high even though its *global* variance
/// is dragged down by the background; a uniformly-soft frame has every tile low, so the quantile
/// stays low and it still flags. Replaces the old whole-image variance, which false-flagged low-key
/// portraits (a sharp face on a dark backdrop) as blurry.
fn tile_peak_focus(gray: &GrayImage) -> f64 {
    let (width, height) = gray.dimensions();
    if width == 0 || height == 0 {
        return 0.0;
    }
    let response = laplacian_response(gray);
    // Fewer pixels than the grid on a side ⇒ tiling is meaningless; fall back to whole-image.
    if width < FOCUS_TILE_GRID || height < FOCUS_TILE_GRID {
        return response_tile_variance(&response, 0, 0, width, height);
    }
    let mut tile_vars: Vec<f64> = Vec::with_capacity((FOCUS_TILE_GRID * FOCUS_TILE_GRID) as usize);
    for ty in 0..FOCUS_TILE_GRID {
        // Integer tile bounds that cover the whole image exactly (the last tile absorbs the slack).
        let y0 = ty * height / FOCUS_TILE_GRID;
        let y1 = (ty + 1) * height / FOCUS_TILE_GRID;
        for tx in 0..FOCUS_TILE_GRID {
            let x0 = tx * width / FOCUS_TILE_GRID;
            let x1 = (tx + 1) * width / FOCUS_TILE_GRID;
            tile_vars.push(response_tile_variance(&response, x0, y0, x1 - x0, y1 - y0));
        }
    }
    percentile(&mut tile_vars, FOCUS_TILE_PERCENTILE)
}

/// The `p`-quantile (`0.0..=1.0`, nearest-rank) of `values`, sorting them in place. Empty ⇒ 0.
fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (p.clamp(0.0, 1.0) * (values.len() as f64 - 1.0)).round() as usize;
    values[rank.min(values.len() - 1)]
}

/// Fraction of pixels sitting in a healthy midtone band — neither crushed toward black nor blown
/// toward white (sc-8563). Distinguishes an intentional low-key frame (a lit subject on a dark
/// backdrop: high `shadow_clip` *and* a healthy well-exposed band) from a genuinely under/over-
/// exposed one (little to nothing in the band). In `[0, 1]`.
fn well_exposed_fraction(gray: &GrayImage) -> f64 {
    let total = u64::from(gray.width()) * u64::from(gray.height());
    if total == 0 {
        return 0.0;
    }
    let healthy = gray
        .pixels()
        .filter(|pixel| (WELL_EXPOSED_LOW..=WELL_EXPOSED_HIGH).contains(&pixel[0]))
        .count() as f64;
    healthy / total as f64
}

/// Fraction of pixels crushed to black and blown to white (luma histogram tails).
fn exposure_clip(gray: &GrayImage) -> (f64, f64) {
    let total = u64::from(gray.width()) * u64::from(gray.height());
    if total == 0 {
        return (0.0, 0.0);
    }
    let mut shadow = 0_u64;
    let mut highlight = 0_u64;
    for pixel in gray.pixels() {
        if pixel[0] <= SHADOW_CUTOFF {
            shadow += 1;
        }
        if pixel[0] >= HIGHLIGHT_CUTOFF {
            highlight += 1;
        }
    }
    (
        shadow as f64 / total as f64,
        highlight as f64 / total as f64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn solid(width: u32, height: u32, color: [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb(color)))
    }

    #[test]
    fn bundled_aesthetic_predictor_loads_and_scores() {
        // The bundled LAION head parses, expects a 768-d CLIP embedding, and scores one to a finite
        // value. (A real-image sanity band is validated end-to-end with the MLX CLIP embedder.)
        let predictor = aesthetic_predictor();
        assert_eq!(
            predictor.input_dim(),
            768,
            "ViT-L/14 image_embeds are 768-d"
        );
        let embedding = vec![0.1_f32; 768];
        let score = predictor.predict(&embedding).expect("finite score");
        assert!(score.is_finite());
        // Same predictor instance is returned (parsed once).
        assert!(std::ptr::eq(predictor, aesthetic_predictor()));
    }

    /// A 1px checkerboard — maximally high-frequency, so its Laplacian variance is large.
    fn checkerboard(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
            if (x + y) % 2 == 0 {
                Rgb([0, 0, 0])
            } else {
                Rgb([255, 255, 255])
            }
        }))
    }

    #[test]
    fn sharp_image_has_higher_blur_variance_than_flat() {
        let flat = scalars_from_image(&solid(64, 64, [128, 128, 128]), 64);
        let sharp = scalars_from_image(&checkerboard(64, 64), 64);
        assert!(flat.blur_variance < 1.0, "flat field responds ~0");
        assert!(
            sharp.blur_variance > flat.blur_variance,
            "checkerboard ({}) should be far sharper than flat ({})",
            sharp.blur_variance,
            flat.blur_variance
        );
    }

    /// A sharp checkerboard patch on an otherwise black frame — the sc-8563 shape (sharp subject,
    /// dark background). Its *global* Laplacian variance is dragged near-zero by the black field, but
    /// tile-peak focus reads the sharp region and stays high. This is the false positive the change
    /// fixes.
    fn sharp_patch_on_black(size: u32, patch: std::ops::Range<u32>) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(size, size, |x, y| {
            let inside = patch.contains(&x) && patch.contains(&y);
            if inside && (x + y) % 2 == 0 {
                Rgb([255, 255, 255])
            } else {
                Rgb([0, 0, 0])
            }
        }))
    }

    #[test]
    fn sharp_region_on_dark_background_is_not_soft() {
        // 48×48 sharp patch (≈3×3 of the 8×8 tiles) on a 128² black frame.
        let subject = scalars_from_image(&sharp_patch_on_black(128, 40..88), 128);
        let all_black = scalars_from_image(&solid(128, 128, [0, 0, 0]), 128);
        assert!(
            subject.blur_variance > 100.0 * all_black.blur_variance.max(1.0),
            "a sharp region on black must read far sharper ({}) than an all-black frame ({})",
            subject.blur_variance,
            all_black.blur_variance
        );
    }

    #[test]
    fn a_single_sharp_tile_does_not_rescue_a_blurry_field() {
        // Only a 16×16 corner (one of the 64 tiles) is sharp; everything else flat. The p90 tile
        // reduction must stay low — one high-frequency block cannot mask an otherwise-soft frame
        // (the false-negative guard for the tile measure).
        let one_tile = scalars_from_image(&sharp_patch_on_black(128, 0..16), 128);
        assert!(
            one_tile.blur_variance < 50.0,
            "a lone sharp tile ({}) must not lift p90 focus above the soft range",
            one_tile.blur_variance
        );
    }

    #[test]
    fn well_exposed_fraction_tracks_healthy_midtones() {
        // Mid-gray is entirely in-band; pure black / pure white are entirely out of band.
        assert!(
            scalars_from_image(&solid(32, 32, [128, 128, 128]), 32).well_exposed_fraction > 0.99
        );
        assert!(scalars_from_image(&solid(32, 32, [0, 0, 0]), 32).well_exposed_fraction < 0.01);
        assert!(
            scalars_from_image(&solid(32, 32, [255, 255, 255]), 32).well_exposed_fraction < 0.01
        );
    }

    /// The sc-8563 acceptance case, end to end (`scalars_from_image` → `evaluate_tier0`): an image
    /// with the measured signature of the real repro — a sharp, well-lit subject on a dark blurred
    /// backdrop — must flag **neither** blur **nor** exposure, and a genuinely-soft version of the
    /// same framing must still flag blur.
    ///
    /// Measured on the actual repro photo (bucket 1024): blurPeak 293.8, shadow_clip 0.315,
    /// well_exposed 0.503 — all three clear their thresholds. The committed fixture is synthetic
    /// (the real photo is third-party/copyrighted) but reproduces that regime: a high-frequency
    /// midtone subject over ~45% clipped-black backdrop.
    #[test]
    fn repro_shape_sharp_subject_on_dark_backdrop_flags_neither() {
        use sceneworks_core::dataset_quality::{
            evaluate_tier0, DatasetKind, ItemQualityInput, QualityCheck, Tier0Thresholds,
        };

        let size = 128;
        let bucket = 128;
        // Left ~55%: a lit subject — high-frequency detail in a healthy midtone band. Right ~45%:
        // the dark backdrop that clips shadows.
        // 4-px blocks (not a 1-px checkerboard, which sits at Nyquist and is low-passed away by the
        // bucket resize); both values are inside the healthy midtone band.
        let sharp_subject_on_black =
            DynamicImage::ImageRgb8(RgbImage::from_fn(size, size, |x, y| {
                if x < 70 {
                    if ((x / 4) + (y / 4)) % 2 == 0 {
                        Rgb([40, 40, 40])
                    } else {
                        Rgb([210, 210, 210])
                    }
                } else {
                    Rgb([0, 0, 0])
                }
            }));
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let to_item = |img: &DynamicImage| ItemQualityInput {
            item_id: "repro".to_owned(),
            width: Some(img.width()),
            height: Some(img.height()),
            content_hash: Some("h".to_owned()),
            scalars: Some(scalars_from_image(img, bucket)),
            acknowledged: Vec::new(),
        };

        let sharp = scalars_from_image(&sharp_subject_on_black, bucket);
        assert!(
            sharp.shadow_clip > thresholds.exposure_clip_fraction,
            "the dark backdrop clips shadows like the repro: {}",
            sharp.shadow_clip
        );
        assert!(
            sharp.well_exposed_fraction > thresholds.well_exposed_min,
            "the lit subject keeps a healthy midtone band: {}",
            sharp.well_exposed_fraction
        );
        assert!(
            sharp.blur_variance > thresholds.blur_floor,
            "the sharp subject clears the blur floor: {}",
            sharp.blur_variance
        );

        let eval = evaluate_tier0(
            std::slice::from_ref(&to_item(&sharp_subject_on_black)),
            bucket,
            1,
            &thresholds,
        );
        let flags = &eval.items[0].flags;
        assert!(
            !flags.iter().any(|f| f.check == QualityCheck::Blur),
            "sharp subject on a dark backdrop must not flag blur (sc-8563)"
        );
        assert!(
            !flags.iter().any(|f| f.check == QualityCheck::Exposure),
            "a lit subject on a clipping backdrop must not flag exposure (sc-8563)"
        );

        // A genuinely blurry version of the SAME image (a Gaussian blur that softens every edge,
        // including the subject/backdrop boundary) ⇒ blur must fire — proving the lowered floor
        // still catches real softness rather than disabling the check. (A hard-edged flat fill would
        // NOT be soft: its subject boundary is itself a sharp edge the peak-tile measure detects,
        // which is exactly the point — genuine blur has no sharp edge anywhere.)
        let blurred = sharp_subject_on_black.blur(3.0);
        let soft = scalars_from_image(&blurred, bucket);
        assert!(
            soft.blur_variance < thresholds.blur_floor,
            "a blurred version reads soft: {}",
            soft.blur_variance
        );
        let eval = evaluate_tier0(
            std::slice::from_ref(&to_item(&blurred)),
            bucket,
            1,
            &thresholds,
        );
        assert!(
            eval.items[0]
                .flags
                .iter()
                .any(|f| f.check == QualityCheck::Blur),
            "a genuinely blurred subject still flags blur"
        );
    }

    #[test]
    fn lit_subject_on_black_keeps_a_healthy_well_exposed_band() {
        // Left half mid-gray (a lit subject), right half black (backdrop): shadows clip ~half the
        // frame, yet ~half sits in the healthy band — the signal that spares a low-key portrait.
        let half = DynamicImage::ImageRgb8(RgbImage::from_fn(64, 64, |x, _| {
            if x < 32 {
                Rgb([128, 128, 128])
            } else {
                Rgb([0, 0, 0])
            }
        }));
        let s = scalars_from_image(&half, 64);
        assert!(
            s.shadow_clip > 0.4,
            "black half clips shadows: {}",
            s.shadow_clip
        );
        assert!(
            s.well_exposed_fraction > 0.4,
            "lit half stays in the healthy band: {}",
            s.well_exposed_fraction
        );
    }

    #[test]
    fn black_image_reads_as_shadow_clipped() {
        let scalars = scalars_from_image(&solid(32, 32, [0, 0, 0]), 32);
        assert!(scalars.shadow_clip > 0.99);
        assert!(scalars.highlight_clip < 0.01);
    }

    #[test]
    fn white_image_reads_as_highlight_clipped() {
        let scalars = scalars_from_image(&solid(32, 32, [255, 255, 255]), 32);
        assert!(scalars.highlight_clip > 0.99);
        assert!(scalars.shadow_clip < 0.01);
    }

    #[test]
    fn midtone_image_is_not_clipped() {
        let scalars = scalars_from_image(&solid(32, 32, [128, 128, 128]), 32);
        assert!(scalars.shadow_clip < 0.01);
        assert!(scalars.highlight_clip < 0.01);
    }

    #[test]
    fn identical_images_hash_identically() {
        let a = scalars_from_image(&checkerboard(48, 48), 64);
        let b = scalars_from_image(&checkerboard(48, 48), 64);
        assert!(!a.phash.is_empty());
        assert_eq!(a.phash, b.phash);
    }

    #[test]
    fn phash_length_is_stable_across_image_sizes() {
        let small = scalars_from_image(&solid(10, 10, [1, 2, 3]), 8);
        let large = scalars_from_image(&checkerboard(100, 40), 64);
        assert_eq!(small.phash.len(), large.phash.len());
    }

    #[test]
    fn extract_from_path_decodes_and_scores() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("checker.png");
        checkerboard(32, 32).save(&path).expect("write png");
        let scalars = extract_tier0_scalars(&path, 32).expect("decode + score");
        assert!(!scalars.phash.is_empty());
        assert!(scalars.blur_variance > 1.0);
    }

    /// A mid-contrast checkerboard: sharp edges (high Laplacian variance) but no exposure clipping
    /// — a clean "good" image, unlike the 0/255 checkerboard which is fully clipped.
    fn mid_checkerboard(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
            if (x + y) % 2 == 0 {
                Rgb([64, 64, 64])
            } else {
                Rgb([192, 192, 192])
            }
        }))
    }

    /// Top 90% black, bottom 10% mid — non-uniform, so it exercises exposure clipping *through* the
    /// Lanczos resize (a solid would be resize-invariant).
    fn mostly_black(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |_, y| {
            if y < height * 9 / 10 {
                Rgb([0, 0, 0])
            } else {
                Rgb([128, 128, 128])
            }
        }))
    }

    /// A strictly-increasing left→right grey ramp (peaks ~200 so a small `offset` never clamps).
    /// Two ramps differing only by `offset` share a gradient/dHash → Hamming 0 → near-duplicates.
    fn gradient(width: u32, height: u32, offset: u8) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, _| {
            let base = (x * 200 / width.max(1)) as u8;
            let v = base.saturating_add(offset);
            Rgb([v, v, v])
        }))
    }

    /// The acceptance check: run *real images* through extract → evaluate and assert the resulting
    /// `QualityFlag`s. This is the only test where the worker-side metric *scale* (real
    /// `blur_variance`) meets the core thresholds (`blur_floor`) — a gross units mismatch would
    /// flag everything or nothing while every isolated test still passed. (Tuning is sc-6530 §8;
    /// this is just sanity that the floor sits between real sharp and soft values.)
    #[test]
    fn extract_then_evaluate_flags_real_images() {
        use sceneworks_core::dataset_quality::{
            evaluate_tier0, DatasetKind, ItemQualityInput, QualityCheck, Tier0Thresholds,
        };

        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let to_item = |id: &str, image: &DynamicImage, content: &str| ItemQualityInput {
            item_id: id.to_owned(),
            width: Some(image.width()),
            height: Some(image.height()),
            content_hash: Some(content.to_owned()),
            scalars: Some(scalars_from_image(image, bucket)),
            acknowledged: Vec::new(),
        };

        let sharp = mid_checkerboard(64, 64);
        let flat = solid(64, 64, [128, 128, 128]);
        let dark = mostly_black(64, 64);
        let ramp_a = gradient(64, 64, 0);
        let ramp_b = gradient(64, 64, 12);

        let items = [
            to_item("sharp", &sharp, "h_sharp"),
            to_item("flat", &flat, "h_flat"),
            to_item("dark", &dark, "h_dark"),
            to_item("ramp_a", &ramp_a, "h_ramp_a"),
            to_item("ramp_b", &ramp_b, "h_ramp_b"),
        ];
        let eval = evaluate_tier0(&items, bucket, 1, &thresholds);

        let flags_of = |id: &str| {
            &eval
                .items
                .iter()
                .find(|entry| entry.item_id == id)
                .expect("item present")
                .flags
        };
        let has = |id: &str, check: QualityCheck| flags_of(id).iter().any(|f| f.check == check);

        // Metric scale: a real sharp image clears the blur floor; a flat field does not.
        assert!(
            !has("sharp", QualityCheck::Blur),
            "a sharp image's real variance must exceed blur_floor"
        );
        assert!(
            has("flat", QualityCheck::Blur),
            "a flat field must read as soft"
        );
        // Exposure clipping survives the Lanczos resize on a non-uniform image.
        assert!(has("dark", QualityCheck::Exposure));
        // Near-duplicate: two ramps differing only in brightness share a dHash.
        let near = flags_of("ramp_a")
            .iter()
            .find(|f| f.check == QualityCheck::NearDuplicate)
            .expect("ramp_a is a near-duplicate of ramp_b");
        assert!(near.peers.contains(&"ramp_b".to_owned()));
    }

    /// sc-8563 threshold calibration probe (not a gate — `#[ignore]`d). Dumps Tier-0 scalars for
    /// every image under `CALIB_DIR` so `blur_floor` + `well_exposed_min` are set from real photos,
    /// not synthetic fixtures, and prints a Gaussian-blur ladder on the sharpest image to show the
    /// sharp→soft drop (the false-negative check on the tile-peak measure).
    ///
    /// `CALIB_DIR=~/Datasets/Kassandra CALIB_BUCKET=1024 \`
    /// `  cargo test -p sceneworks-image-quality calibrate_tier0 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn calibrate_tier0_over_real_images() {
        let Ok(dir) = std::env::var("CALIB_DIR") else {
            eprintln!("set CALIB_DIR to a directory of images");
            return;
        };
        let bucket: u32 = std::env::var("CALIB_BUCKET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);

        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, out);
                } else if matches!(
                    path.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.to_ascii_lowercase())
                        .as_deref(),
                    Some("png" | "jpg" | "jpeg" | "webp")
                ) {
                    out.push(path);
                }
            }
        }

        let mut paths = Vec::new();
        walk(Path::new(&dir), &mut paths);
        paths.sort();

        let mut rows: Vec<(String, Tier0Scalars)> = Vec::new();
        for path in &paths {
            if let Ok(scalars) = extract_tier0_scalars(path, bucket) {
                rows.push((path.display().to_string(), scalars));
            }
        }
        rows.sort_by(|a, b| {
            a.1.blur_variance
                .partial_cmp(&b.1.blur_variance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        eprintln!("bucket={bucket}  n={}", rows.len());
        eprintln!(
            "{:>12}  {:>7}  {:>7}  {:>8}  path",
            "blurPeak", "shadow", "highl", "wellExp"
        );
        for (path, s) in &rows {
            eprintln!(
                "{:12.1}  {:7.3}  {:7.3}  {:8.3}  {}",
                s.blur_variance, s.shadow_clip, s.highlight_clip, s.well_exposed_fraction, path
            );
        }

        if let Some((path, _)) = rows.last() {
            if let Ok(img) = image::open(path) {
                eprintln!("--- gaussian blur ladder on sharpest ({path}) ---");
                for sigma in [0.0_f32, 0.5, 1.0, 2.0, 4.0, 8.0] {
                    let candidate = if sigma == 0.0 {
                        img.clone()
                    } else {
                        img.blur(sigma)
                    };
                    let s = scalars_from_image(&candidate, bucket);
                    eprintln!(
                        "sigma {sigma:>4.1}  blurPeak {:12.1}  wellExp {:.3}",
                        s.blur_variance, s.well_exposed_fraction
                    );
                }
            }
        }
    }

    #[test]
    fn compute_readiness_extracts_evaluates_and_reports() {
        use sceneworks_core::dataset_quality::{
            DatasetKind, QualityCheck, ReadinessGate, Tier0Thresholds,
        };

        let dir = tempfile::tempdir().expect("temp dir");
        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);

        let sharp_path = dir.path().join("sharp.png");
        mid_checkerboard(64, 64)
            .save(&sharp_path)
            .expect("write sharp");
        let flat_path = dir.path().join("flat.png");
        solid(64, 64, [128, 128, 128])
            .save(&flat_path)
            .expect("write flat");

        let items = vec![
            ReadinessItem {
                item_id: "sharp".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("h_sharp".to_owned()),
                image_path: Some(sharp_path),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
            ReadinessItem {
                item_id: "flat".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("h_flat".to_owned()),
                image_path: Some(flat_path),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
        ];

        let (report, extracted) =
            compute_readiness(&items, bucket, 1, &thresholds, None, None, None, None);

        // Both decoded fresh (returned for the caller to cache); flat reads soft → NeedsAttention.
        assert_eq!(extracted.len(), 2);
        assert_eq!(report.gate, ReadinessGate::NeedsAttention);
        let flat = report
            .items
            .iter()
            .find(|i| i.item_id == "flat")
            .expect("flat");
        assert!(flat.flags.iter().any(|f| f.check == QualityCheck::Blur));
    }

    #[test]
    fn path_can_carry_metadata_is_true_for_everything_but_png() {
        use std::path::Path;
        assert!(path_can_carry_metadata(Path::new("a/b/photo.jpg")));
        assert!(path_can_carry_metadata(Path::new("photo.JPEG")));
        assert!(path_can_carry_metadata(Path::new("photo.heic")));
        assert!(path_can_carry_metadata(Path::new("noext")));
        assert!(!path_can_carry_metadata(Path::new("clean.png")));
        assert!(!path_can_carry_metadata(Path::new("clean.PNG")));
    }

    #[test]
    fn compute_readiness_marks_only_non_png_items_as_metadata_strippable() {
        use sceneworks_core::dataset_quality::{DatasetKind, Tier0Thresholds};

        let dir = tempfile::tempdir().expect("temp dir");
        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);

        // A JPEG still carries EXIF/GPS/ICC; a PNG is the normalized, metadata-free result of a strip.
        let jpg_path = dir.path().join("photo.jpg");
        mid_checkerboard(64, 64).save(&jpg_path).expect("write jpg");
        let png_path = dir.path().join("clean.png");
        mid_checkerboard(64, 64).save(&png_path).expect("write png");

        let items = vec![
            ReadinessItem {
                item_id: "jpg".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("h_jpg".to_owned()),
                image_path: Some(jpg_path),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
            ReadinessItem {
                item_id: "png".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("h_png".to_owned()),
                image_path: Some(png_path),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
        ];

        let (report, _) = compute_readiness(&items, bucket, 1, &thresholds, None, None, None, None);

        let strippable: std::collections::HashMap<_, _> = report
            .items
            .iter()
            .map(|i| (i.item_id.as_str(), i.metadata_strippable))
            .collect();
        assert_eq!(strippable.get("jpg"), Some(&true));
        assert_eq!(strippable.get("png"), Some(&false));
    }

    #[test]
    fn compute_readiness_plans_dropping_exact_duplicates_keeping_sharpest() {
        use sceneworks_core::dataset_quality::{DatasetKind, Tier0Thresholds};

        // Two byte-identical copies (same content hash + pHash) → exact duplicates; the sharper one is
        // kept. cached_scalars lets the test set sharpness without writing image files.
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let dup = |id: &str, blur: f64| ReadinessItem {
            item_id: id.to_owned(),
            width: Some(64),
            height: Some(64),
            content_hash: Some("same".to_owned()),
            image_path: None,
            cached_scalars: Some(Tier0Scalars {
                blur_variance: blur,
                shadow_clip: 0.0,
                highlight_clip: 0.0,
                well_exposed_fraction: 1.0,
                phash: vec![0, 0, 0, 0, 0, 0, 0, 0],
            }),
            acknowledged: Vec::new(),
        };
        let items = vec![dup("soft", 5000.0), dup("sharp", 9000.0)];

        let (report, _) = compute_readiness(&items, 64, 1, &thresholds, None, None, None, None);
        let plan = report
            .duplicate_removal
            .expect("exact duplicates produce a removal plan");
        assert_eq!(plan.groups.len(), 1);
        assert_eq!(plan.groups[0].keep, "sharp");
        assert_eq!(plan.groups[0].remove, vec!["soft".to_owned()]);
    }

    #[test]
    fn compute_readiness_does_not_auto_remove_clip_near_duplicates() {
        use sceneworks_core::dataset_quality::{
            evaluate_tier1, DatasetKind, ItemEmbedding, Tier0Thresholds, Tier1Thresholds,
        };

        // Two distinct files (different content hash + far-apart pHash, so NO Tier-0 duplicate), but
        // identical CLIP embeddings → a NearDuplicateEmbedding. That is legitimate training variety,
        // so it must surface as a flag yet NEVER become a one-tap auto-remove (it would fight the
        // sibling LowDiversity check).
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let clean = |id: &str, phash: Vec<u8>, hash: &str| ReadinessItem {
            item_id: id.to_owned(),
            width: Some(64),
            height: Some(64),
            content_hash: Some(hash.to_owned()),
            image_path: None,
            cached_scalars: Some(Tier0Scalars {
                blur_variance: 5000.0,
                shadow_clip: 0.0,
                highlight_clip: 0.0,
                well_exposed_fraction: 1.0,
                phash,
            }),
            acknowledged: Vec::new(),
        };
        let items = vec![
            clean("a", vec![0, 0, 0, 0, 0, 0, 0, 0], "ha"),
            clean("b", vec![255, 255, 255, 255, 255, 255, 255, 255], "hb"),
        ];
        let embedding = |id: &str| ItemEmbedding {
            item_id: id.to_owned(),
            embedding: vec![1.0, 0.0],
            acknowledged: Vec::new(),
        };
        let tier1 = evaluate_tier1(
            &[embedding("a"), embedding("b")],
            &Tier1Thresholds::for_kind(&DatasetKind::Person),
        );

        let (report, _) =
            compute_readiness(&items, 64, 1, &thresholds, Some(&tier1), None, None, None);
        assert!(
            report
                .items
                .iter()
                .flat_map(|item| &item.flags)
                .any(|flag| flag.check == QualityCheck::NearDuplicateEmbedding),
            "the CLIP near-duplicate is still surfaced as a flag"
        );
        assert!(
            report.duplicate_removal.is_none(),
            "CLIP near-duplicates must not become an auto-remove plan"
        );
    }

    #[test]
    fn compute_readiness_reuses_cache_and_flags_decode_failure() {
        use sceneworks_core::dataset_quality::{DatasetKind, QualityCheck, Tier0Thresholds};

        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let cached = scalars_from_image(&mid_checkerboard(64, 64), bucket);

        let items = vec![
            ReadinessItem {
                item_id: "cached".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("h".to_owned()),
                image_path: None, // no path → must use the cache
                cached_scalars: Some(cached),
                acknowledged: Vec::new(),
            },
            ReadinessItem {
                item_id: "broken".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("h_broken".to_owned()),
                image_path: Some(PathBuf::from("/no/such/file.png")),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
        ];

        let (report, extracted) =
            compute_readiness(&items, bucket, 1, &thresholds, None, None, None, None);

        // The cached item is not re-extracted; the broken path yields nothing to cache.
        assert!(extracted.is_empty());
        let broken = report
            .items
            .iter()
            .find(|i| i.item_id == "broken")
            .expect("broken");
        assert!(broken.flags.iter().any(|f| f.check == QualityCheck::Decode));
    }

    #[test]
    fn acknowledging_a_finding_drops_it_from_the_rollup() {
        use sceneworks_core::dataset_quality::{
            DatasetKind, QualityCheck, ReadinessGate, Tier0Thresholds,
        };

        let dir = tempfile::tempdir().expect("temp dir");
        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let flat_path = dir.path().join("flat.png");
        solid(64, 64, [128, 128, 128])
            .save(&flat_path)
            .expect("write flat");

        let items = vec![ReadinessItem {
            item_id: "flat".to_owned(),
            width: Some(64),
            height: Some(64),
            content_hash: Some("h_flat".to_owned()),
            image_path: Some(flat_path),
            cached_scalars: None,
            acknowledged: vec![QualityCheck::Blur],
        }];

        let (report, _) = compute_readiness(&items, bucket, 1, &thresholds, None, None, None, None);
        // Blur is the only finding and the user dismissed it → Ready, badge clean…
        assert_eq!(report.gate, ReadinessGate::Ready);
        assert_eq!(report.items[0].severity, None);
        // …but the flag is still in the payload, marked acknowledged for the struck-through display.
        let blur = report.items[0]
            .flags
            .iter()
            .find(|f| f.check == QualityCheck::Blur)
            .expect("blur kept");
        assert!(blur.acknowledged);
    }

    #[test]
    fn decode_failure_cannot_be_acknowledged() {
        use sceneworks_core::dataset_quality::{
            DatasetKind, QualityCheck, ReadinessGate, Tier0Thresholds,
        };

        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let items = vec![ReadinessItem {
            item_id: "broken".to_owned(),
            width: Some(64),
            height: Some(64),
            content_hash: Some("h_broken".to_owned()),
            image_path: Some(PathBuf::from("/no/such/file.png")),
            cached_scalars: None,
            acknowledged: vec![QualityCheck::Decode], // even if asked, a decode failure stands
        }];

        let (report, _) = compute_readiness(&items, bucket, 1, &thresholds, None, None, None, None);
        let decode = report.items[0]
            .flags
            .iter()
            .find(|f| f.check == QualityCheck::Decode)
            .expect("decode flag");
        assert!(!decode.acknowledged);
        assert_ne!(report.gate, ReadinessGate::Ready);
    }

    #[test]
    fn compute_readiness_emits_per_metric_distributions() {
        use sceneworks_core::dataset_quality::{DatasetKind, Tier0Thresholds};

        let dir = tempfile::tempdir().expect("temp dir");
        let bucket = 64;
        let thresholds = Tier0Thresholds::for_kind(&DatasetKind::Person);
        let a_path = dir.path().join("a.png");
        mid_checkerboard(64, 64).save(&a_path).expect("write a");
        let b_path = dir.path().join("b.png");
        solid(64, 64, [128, 128, 128])
            .save(&b_path)
            .expect("write b");

        let items = vec![
            ReadinessItem {
                item_id: "a".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("ha".to_owned()),
                image_path: Some(a_path),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
            ReadinessItem {
                item_id: "b".to_owned(),
                width: Some(64),
                height: Some(64),
                content_hash: Some("hb".to_owned()),
                image_path: Some(b_path),
                cached_scalars: None,
                acknowledged: Vec::new(),
            },
        ];

        let (report, _) = compute_readiness(&items, bucket, 1, &thresholds, None, None, None, None);
        let dist = report.distributions.expect("distributions present");
        // One value per decodable item, oriented + thresholded for the chart.
        assert_eq!(dist.blur_variance.values.len(), 2);
        assert!(dist.blur_variance.higher_is_better);
        assert_eq!(dist.blur_variance.threshold, Some(thresholds.blur_floor));
        assert!(!dist.shadow_clip.higher_is_better);
        assert_eq!(dist.shadow_clip.values.len(), 2);
    }
}
