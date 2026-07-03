//! Backend-neutral SAM3 person-segmentation helpers shared by the two cfg-exclusive SAM3 modules —
//! the native-MLX [`crate::person_segment_sam3`] (macOS) and the candle
//! [`crate::person_segment_sam3_candle`] (off-Mac / `backend-candle`) (sc-8847, F-045).
//!
//! Both modules drive the *same* SAM3 PCS video pipeline and differ ONLY in the tensor/model/device
//! seam (`mlx_rs::Array` + `mlx-gen-sam3` vs `candle_core::Tensor` + `candle-gen-sam3`). The weight
//! resolution, RGB→CHW normalization, and the mask/association MATH — the correctness core of
//! Replace-Person and the SCAIL-2 painter — carry no tensor dependency and were previously duplicated
//! VERBATIM across the two files (~300 lines). They live here ONCE so a fix lands on both platforms at
//! once, and the two backends can no longer silently diverge.
//!
//! Compiled on BOTH lanes via the same superset gate as [`crate::scail2_masks`]
//! (`any(macos, all(not(macos), backend-candle))`); the per-backend files keep only their tensor seam.

use std::path::{Path, PathBuf};

use crate::downloads::{ensure_hf_cached_file, DownloadContext};
use crate::{Settings, WorkerResult};

/// SAM3 checkpoint files (loaded stock from `facebook/sam3`, no conversion).
pub(crate) const MODEL_FILE: &str = "model.safetensors";
pub(crate) const TOKENIZER_FILE: &str = "tokenizer.json";

/// Download-on-first-use repo: the stock `facebook/sam3` checkpoint mirror owned by SceneWorks (the
/// same `model.safetensors` + `tokenizer.json` both backends use — no MLX-specific conversion, despite
/// the `-mlx` name). Publishing the public mirror is gated on sign-off (Meta SAM License → must ship a
/// LICENSE copy); until then point `SCENEWORKS_SAM3_WEIGHTS` at a local `facebook/sam3` snapshot dir.
pub(crate) const SEG_REPO: &str = "SceneWorks/sam3-mlx";

/// SAM3 model input is a square 1008×1008 (the processor resizes to a fixed square, not
/// aspect-preserving — `Sam3VideoProcessor` `size={1008,1008}`, `default_to_square`).
pub(crate) const INPUT_SIZE: u32 = 1008;

/// SAM3 mask logits come back on a 288×288 low-res grid (`Sam3VideoModel` `LOW_RES`).
pub(crate) const MASK_GRID: usize = 288;

/// The text concept driving PCS — the whole point of the SAM3 upgrade (no manual box).
pub(crate) const CONCEPT_PROMPT: &str = "person";

/// A normalized `(x, y, width, height)` box (0..1), the per-frame ByteTrack anchor — used only to
/// *associate* a SAM3 object id with the selected track, never as a segmenter prompt.
pub(crate) type BoxNorm = (f64, f64, f64, f64);

/// Backend-neutral view of a SAM3 per-frame output (`obj_id → 288² mask logits`), letting the shared
/// association math run over either backend's `VideoFrameOutput` without depending on either sam3
/// crate. Both `mlx-gen-sam3::VideoFrameOutput` and `candle-gen-sam3::VideoFrameOutput` are the same
/// shape (`{ obj_ids: Vec<i32>, masks: Vec<Vec<f32>> }`); each per-backend module impls this trait on
/// its own type so [`select_object`] stays a single copy of the selection logic.
pub(crate) trait Sam3FrameOutput {
    /// SAM3 object ids present on this frame, parallel to [`Sam3FrameOutput::masks`].
    fn obj_ids(&self) -> &[i32];
    /// Per-object 288² low-res mask logits, parallel to [`Sam3FrameOutput::obj_ids`].
    fn masks(&self) -> &[Vec<f32>];
}

/// Resolve already-present SAM3 weights: an explicit env pin (`SCENEWORKS_SAM3_WEIGHTS`, a dir or
/// the `model.safetensors` inside it), then the app cache `<data_dir>/cache/person-segment-sam3/`,
/// then the model dir `<data_dir>/models/person-segment-sam3/`. Both `model.safetensors` and
/// `tokenizer.json` must be present. Returns `(model_path, tokenizer_path)` or `None` (then
/// [`ensure_segmenter_weights`] downloads them).
pub(crate) fn resolve_segmenter_weights(settings: &Settings) -> Option<(PathBuf, PathBuf)> {
    let pair_in = |dir: &Path| -> Option<(PathBuf, PathBuf)> {
        let model = dir.join(MODEL_FILE);
        let tokenizer = dir.join(TOKENIZER_FILE);
        (model.exists() && tokenizer.exists()).then_some((model, tokenizer))
    };
    if let Ok(pinned) = std::env::var("SCENEWORKS_SAM3_WEIGHTS") {
        let p = PathBuf::from(pinned);
        let dir = if p.is_file() {
            p.parent().map(Path::to_path_buf).unwrap_or(p)
        } else {
            p
        };
        if let Some(pair) = pair_in(&dir) {
            return Some(pair);
        }
    }
    for sub in ["cache/person-segment-sam3", "models/person-segment-sam3"] {
        if let Some(pair) = pair_in(&settings.data_dir.join(sub)) {
            return Some(pair);
        }
    }
    None
}

/// Resolve the SAM3 weights, downloading `model.safetensors` + `tokenizer.json` from HuggingFace
/// on first use (into the app cache) with streaming progress/cancel and size-aware resume.
pub(crate) async fn ensure_segmenter_weights(
    settings: &Settings,
    context: &DownloadContext<'_>,
) -> WorkerResult<(PathBuf, PathBuf)> {
    if let Some(pair) = resolve_segmenter_weights(settings) {
        return Ok(pair);
    }
    let dir = settings.data_dir.join("cache").join("person-segment-sam3");
    let model =
        ensure_hf_cached_file(context, SEG_REPO, "main", MODEL_FILE, &dir.join(MODEL_FILE)).await?;
    let tokenizer = ensure_hf_cached_file(
        context,
        SEG_REPO,
        "main",
        TOKENIZER_FILE,
        &dir.join(TOKENIZER_FILE),
    )
    .await?;
    Ok((model, tokenizer))
}

/// Pack a `size×size` interleaved-RGB `u8` buffer into a channel-major `[3·size·size]` f32 vector
/// with the SAM3 normalization `(x/255 − 0.5) / 0.5` = `x/127.5 − 1` (mean=std=0.5 → range
/// `[-1,1]`). Split out so the normalization is unit-testable without an image decode; the per-backend
/// `input_tensor` wraps this in its `Array`/`Tensor`.
pub(crate) fn normalize_chw(rgb: &[u8], size: usize) -> Vec<f32> {
    let plane = size * size;
    let mut out = vec![0f32; 3 * plane];
    for (p, px) in rgb.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * plane + p] = px[c] as f32 / 127.5 - 1.0;
        }
    }
    out
}

/// Fraction of a SAM3 object's foreground (mask logit `> 0` ⇔ σ `> 0.5`) on the `grid×grid`
/// low-res mask whose pixel center falls inside the normalized `box_norm`. `0.0` when the mask is
/// empty. SAM3 masks live in the squashed 1008² space, which maps to the full normalized frame, so
/// a mask pixel `(mx,my)` has normalized center `((mx+0.5)/grid, (my+0.5)/grid)` — directly
/// comparable to the ByteTrack box. This is the association score: how much of a candidate object
/// sits under the selected person's box.
pub(crate) fn mask_box_containment(mask_logits: &[f32], grid: usize, box_norm: BoxNorm) -> f64 {
    let (bx, by, bw, bh) = box_norm;
    let (x1, y1, x2, y2) = (bx, by, bx + bw, by + bh);
    let (mut fg, mut inside) = (0u64, 0u64);
    for my in 0..grid {
        for mx in 0..grid {
            if mask_logits[my * grid + mx] > 0.0 {
                fg += 1;
                let (cx, cy) = (
                    (mx as f64 + 0.5) / grid as f64,
                    (my as f64 + 0.5) / grid as f64,
                );
                if cx >= x1 && cx < x2 && cy >= y1 && cy < y2 {
                    inside += 1;
                }
            }
        }
    }
    if fg == 0 {
        0.0
    } else {
        inside as f64 / fg as f64
    }
}

/// Pick the SAM3 object id that best matches the selected track: for every clip frame with an
/// anchor box, accumulate each present object's [`mask_box_containment`], then take the id with the
/// greatest total. `None` when no object ever overlaps an anchor (→ degraded to box masks). The
/// per-frame sum (not a single best frame) rewards an object that stays under the box across the
/// span, which disambiguates the selected person from nearby people SAM3 also segmented. Generic over
/// [`Sam3FrameOutput`] so the identical selection math runs on either backend's frame type.
pub(crate) fn select_object<F: Sam3FrameOutput>(
    outputs: &[F],
    anchors: &[Option<BoxNorm>],
) -> Option<i32> {
    use std::collections::HashMap;
    let mut score: HashMap<i32, f64> = HashMap::new();
    for (frame, anchor) in outputs.iter().zip(anchors) {
        let Some(box_norm) = anchor else { continue };
        for (oid, mask) in frame.obj_ids().iter().zip(frame.masks()) {
            let c = mask_box_containment(mask, MASK_GRID, *box_norm);
            if c > 0.0 {
                *score.entry(*oid).or_insert(0.0) += c;
            }
        }
    }
    // Deterministic tie-break on the object id (BTree-like) so repeated runs agree.
    score
        .into_iter()
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.cmp(&a.0))
        })
        .map(|(oid, _)| oid)
}

/// Binarize a `grid×grid` SAM3 mask (logit `> 0`) to a 0/255 buffer, then resize it to
/// `width×height` (bilinear) and re-threshold to a clean binary `L` mask — the per-clip-frame
/// output the orchestrator writes. Inverts SAM3's uniform 1008² squash back to the frame aspect.
pub(crate) fn mask_to_frame(mask_logits: &[f32], grid: usize, width: u32, height: u32) -> Vec<u8> {
    let bin: Vec<u8> = mask_logits
        .iter()
        .map(|&v| if v > 0.0 { 255 } else { 0 })
        .collect();
    let Some(small) = image::GrayImage::from_raw(grid as u32, grid as u32, bin) else {
        return Vec::new();
    };
    let resized =
        image::imageops::resize(&small, width, height, image::imageops::FilterType::Triangle);
    resized
        .into_raw()
        .into_iter()
        .map(|v| if v > 127 { 255 } else { 0 })
        .collect()
}

/// Normalized centroid-x (0..1) of a SAM3 low-res mask's foreground (logit `> 0`); `None` if the
/// mask is empty. Used to sort tracked people left-to-right for deterministic palette assignment.
pub(crate) fn mask_centroid_x(mask_logits: &[f32], grid: usize) -> Option<f64> {
    let (mut sum_x, mut n) = (0f64, 0u64);
    for my in 0..grid {
        for mx in 0..grid {
            if mask_logits[my * grid + mx] > 0.0 {
                sum_x += (mx as f64 + 0.5) / grid as f64;
                n += 1;
            }
        }
    }
    (n > 0).then(|| sum_x / n as f64)
}

/// Every tracked person's per-frame mask + a stable left-to-right paint order — the input to the
/// SCAIL-2 color-mask painter (sc-5448). Backend-neutral (pure mask bytes): both SAM3 modules build
/// this and [`crate::scail2_masks`]'s painters consume it. Unlike the per-backend `segment_track`
/// (which selects ONE person via ByteTrack anchors for replace_person), this keeps EVERY SAM3 object
/// so each person can be painted a distinct palette color.
pub(crate) struct AllPersonMasks {
    /// SAM3 object ids in left-to-right paint order (ascending centroid-x in the frame where each
    /// object first appears); person index 0 = leftmost = the SCAIL-2 palette's first color.
    pub order: Vec<i32>,
    /// `per_frame[f]` = `(obj_id, binary mask row-major width*height, 0/255)` for every object
    /// present on frame `f` (empty masks dropped). Object ids index into [`AllPersonMasks::order`].
    pub per_frame: Vec<Vec<(i32, Vec<u8>)>>,
    pub width: u32,
    pub height: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny backend-neutral stand-in for either sam3 crate's `VideoFrameOutput`, so the shared
    /// [`select_object`] math is tested here (compiled on both lanes) without depending on
    /// `mlx-gen-sam3` / `candle-gen-sam3`.
    struct TestFrame {
        obj_ids: Vec<i32>,
        masks: Vec<Vec<f32>>,
    }
    impl Sam3FrameOutput for TestFrame {
        fn obj_ids(&self) -> &[i32] {
            &self.obj_ids
        }
        fn masks(&self) -> &[Vec<f32>] {
            &self.masks
        }
    }

    #[test]
    fn normalize_maps_to_signed_unit_range_channel_major() {
        // 2×2 RGB: black, white, and a mid-gray pixel. mean=std=0.5 → x/127.5 − 1.
        let rgb = [
            0, 0, 0, // (0,0) black
            255, 255, 255, // (0,1) white
            128, 128, 128, // (1,0) ~mid
            255, 0, 0, // (1,1) red
        ];
        let chw = normalize_chw(&rgb, 2);
        let plane = 4;
        // Channel-major: R plane first.
        assert!((chw[0] - (-1.0)).abs() < 1e-6); // R of black
        assert!((chw[1] - 1.0).abs() < 1e-6); // R of white
        assert!((chw[plane] - (-1.0)).abs() < 1e-6); // G of black
        assert!((chw[2 * plane + 3] - (-1.0)).abs() < 1e-6); // B of red pixel = 0 → -1
        assert!((chw[3] - 1.0).abs() < 1e-6); // R of red pixel = 255 → 1
    }

    #[test]
    fn containment_measures_foreground_inside_box() {
        // 10×10 grid, a 4×4 foreground block at rows/cols 2..6 (logit 1.0 inside, -1.0 outside).
        let grid = 10;
        let mut logits = vec![-1.0f32; grid * grid];
        for my in 2..6 {
            for mx in 2..6 {
                logits[my * grid + mx] = 1.0;
            }
        }
        // Box covering the whole block (normalized 0.2..0.6) → containment 1.0.
        let full = mask_box_containment(&logits, grid, (0.2, 0.2, 0.4, 0.4));
        assert!((full - 1.0).abs() < 1e-9, "full was {full}");
        // Box over the empty top-left corner → 0.0.
        let none = mask_box_containment(&logits, grid, (0.0, 0.0, 0.1, 0.1));
        assert!(none.abs() < 1e-9, "disjoint was {none}");
        // Box over the left half of the block → ~half the foreground inside.
        let half = mask_box_containment(&logits, grid, (0.0, 0.0, 0.4, 1.0));
        assert!((half - 0.5).abs() < 1e-9, "half was {half}");
        // Empty mask → 0.0, never divide-by-zero.
        assert_eq!(
            mask_box_containment(&vec![-1.0; grid * grid], grid, (0.0, 0.0, 1.0, 1.0)),
            0.0
        );
    }

    /// A small synthetic two-object clip: object 7 sits under the anchor box every frame, object 9
    /// sits elsewhere. The selector must return 7, aggregating containment across the span.
    #[test]
    fn select_object_picks_the_id_overlapping_the_anchor() {
        let grid = MASK_GRID;
        let block = |r: std::ops::Range<usize>, c: std::ops::Range<usize>| -> Vec<f32> {
            let mut m = vec![-1.0f32; grid * grid];
            for y in r.clone() {
                for x in c.clone() {
                    m[y * grid + x] = 1.0;
                }
            }
            m
        };
        // obj 7 in the top-left quadrant; obj 9 in the bottom-right.
        let left = block(0..grid / 2, 0..grid / 2);
        let right = block(grid / 2..grid, grid / 2..grid);
        let outputs = vec![
            TestFrame {
                obj_ids: vec![7, 9],
                masks: vec![left.clone(), right.clone()],
            },
            TestFrame {
                obj_ids: vec![7, 9],
                masks: vec![left, right],
            },
        ];
        // Anchor over the top-left quadrant on both frames.
        let anchors = vec![Some((0.0, 0.0, 0.5, 0.5)), Some((0.0, 0.0, 0.5, 0.5))];
        assert_eq!(select_object(&outputs, &anchors), Some(7));
        // No anchors → no selection.
        assert_eq!(select_object(&outputs, &[None, None]), None);
    }

    #[test]
    fn mask_to_frame_binarizes_and_resizes() {
        // 4×4 grid, top-left 2×2 foreground → resized to 8×8 should keep a top-left fg region.
        let grid = 4;
        let mut logits = vec![-1.0f32; grid * grid];
        for y in 0..2 {
            for x in 0..2 {
                logits[y * grid + x] = 1.0;
            }
        }
        let out = mask_to_frame(&logits, grid, 8, 8);
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|&v| v == 0 || v == 255), "output is binary");
        assert_eq!(out[0], 255, "top-left corner is foreground");
        assert_eq!(out[63], 0, "bottom-right corner is background");
    }
}
