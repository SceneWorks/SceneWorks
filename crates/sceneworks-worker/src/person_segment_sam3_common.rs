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

use gen_core::CancelFlag;

use crate::downloads::{ensure_hf_cached_file, DownloadContext};
use crate::{Settings, WorkerError, WorkerResult};

/// Cancel copy surfaced when a SAM2/SAM3 person segmentation is interrupted by a user cancel — either
/// by an engine's per-frame propagate cancel contract (gen-core d8038beb / sc-8972) or by the coarse
/// checks around the cold weight load (sc-8807). Shared by the SAM2 module ([`crate::person_segment`])
/// and both SAM3 twins (sc-11191, F-018): the candle twin is off-Mac, so it cannot reach the Mac-only
/// `person_segment`; hoisting the trio here gives both cfg-exclusive twins ONE definition and stops a
/// tweak in one silently diverging from the other.
pub(crate) const CANCEL_MESSAGE: &str = "Person segmentation canceled by user.";

/// Per-propagated-frame progress callback `(frame_index, total_frames)`, invoked from the blocking
/// thread after each propagated frame (the gen-core d8038beb video per-step progress contract).
/// Boxed + `Send` so call sites can move it into `spawn_blocking`. Shared by the SAM2/SAM3 modules
/// (sc-11191, F-018).
pub(crate) type SegmentProgress = Box<dyn FnMut(usize, usize) + Send>;

/// Bail out with [`WorkerError::Canceled`] when the threaded flag has been tripped — the coarse
/// cancel seam guarding the phases the engine cannot observe (frame decode, the cold multi-GB weight
/// load/parse, quantize). The engine itself checks the same flag between frames (sc-8807). Shared by
/// the SAM2/SAM3 modules (sc-11191, F-018).
pub(crate) fn check_segment_canceled(cancel: Option<&CancelFlag>) -> WorkerResult<()> {
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    Ok(())
}

/// SAM3 checkpoint files (loaded stock from `facebook/sam3`, no conversion).
pub(crate) const MODEL_FILE: &str = "model.safetensors";
pub(crate) const TOKENIZER_FILE: &str = "tokenizer.json";

/// Download-on-first-use repo: the stock `facebook/sam3` checkpoint mirror owned by SceneWorks (the
/// same `model.safetensors` + `tokenizer.json` both backends use — no MLX-specific conversion, despite
/// the `-mlx` name). Publishing the public mirror is gated on sign-off (Meta SAM License → must ship a
/// LICENSE copy); until then point `SCENEWORKS_SAM3_WEIGHTS` at a local `facebook/sam3` snapshot dir.
pub(crate) const SEG_REPO: &str = "SceneWorks/sam3-mlx";

/// Pinned SAM3 weights revision (sc-9879, F-077 follow-up). Even though `SEG_REPO` is a
/// first-party repo, fetching the mutable `main` branch means a re-push (or a compromised
/// token) could silently swap the `model.safetensors` / `tokenizer.json` we load. Pin the
/// exact commit for defense-in-depth, mirroring the person-detector pin (sc-9682). HF's tree
/// API still reports each file's `lfs.oid`, which `ensure_hf_cached_file` verifies against.
pub(crate) const SEG_REVISION: &str = "3ed10b164a755c00b1a4d671dde95719c127e1a7";

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
/// `tokenizer.json` must be present. Returns `Ok(Some((model_path, tokenizer_path)))` or `Ok(None)`
/// (then [`ensure_segmenter_weights`] downloads them).
///
/// A set-but-missing `SCENEWORKS_SAM3_WEIGHTS` is an operator error: it fails loudly
/// (`InvalidPayload`) instead of silently falling through to the cache/HF download and loading
/// different weights than the operator asked for (sc-11175/F-011, mirroring the sc-8911 upscaler
/// pin). [`crate::util::resolve_env_file_pin`] errors if the pinned path itself is absent; an
/// incomplete pinned dir (missing either file of the pair) is likewise a loud error, not a
/// silent fall-through.
pub(crate) fn resolve_segmenter_weights(
    settings: &Settings,
) -> WorkerResult<Option<(PathBuf, PathBuf)>> {
    let pair_in = |dir: &Path| -> Option<(PathBuf, PathBuf)> {
        let model = dir.join(MODEL_FILE);
        let tokenizer = dir.join(TOKENIZER_FILE);
        (model.exists() && tokenizer.exists()).then_some((model, tokenizer))
    };
    if let Some(pinned) = crate::util::resolve_env_file_pin(
        "SCENEWORKS_SAM3_WEIGHTS",
        std::env::var_os("SCENEWORKS_SAM3_WEIGHTS"),
        "a local facebook/sam3 snapshot dir (or its model.safetensors)",
    )? {
        let dir = if pinned.is_file() {
            pinned.parent().map(Path::to_path_buf).unwrap_or(pinned)
        } else {
            pinned
        };
        return pair_in(&dir).map(Some).ok_or_else(|| {
            crate::WorkerError::InvalidPayload(format!(
                "SCENEWORKS_SAM3_WEIGHTS is set to {} but that directory is missing {MODEL_FILE} \
                 and/or {TOKENIZER_FILE}. Point it at a complete facebook/sam3 snapshot dir, or \
                 unset it to download on first use.",
                dir.display()
            ))
        });
    }
    for sub in ["cache/person-segment-sam3", "models/person-segment-sam3"] {
        if let Some(pair) = pair_in(&settings.data_dir.join(sub)) {
            return Ok(Some(pair));
        }
    }
    Ok(None)
}

/// Resolve the SAM3 weights, downloading `model.safetensors` + `tokenizer.json` from HuggingFace
/// on first use (into the app cache) with streaming progress/cancel and size-aware resume.
pub(crate) async fn ensure_segmenter_weights(
    settings: &Settings,
    context: &DownloadContext<'_>,
) -> WorkerResult<(PathBuf, PathBuf)> {
    if let Some(pair) = resolve_segmenter_weights(settings)? {
        return Ok(pair);
    }
    let dir = settings.data_dir.join("cache").join("person-segment-sam3");
    let model = ensure_hf_cached_file(
        context,
        SEG_REPO,
        SEG_REVISION,
        MODEL_FILE,
        &dir.join(MODEL_FILE),
    )
    .await?;
    let tokenizer = ensure_hf_cached_file(
        context,
        SEG_REPO,
        SEG_REVISION,
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
///
/// Returns an `Engine` error when the logit buffer's length doesn't match `grid×grid` (so
/// `GrayImage::from_raw` would fail): the old code returned an empty vec there, which is the
/// exact sentinel the orchestrator reads as "object absent → box fallback" — so a malformed
/// mask silently degraded to no-mask instead of surfacing (sc-8905, F-103). Legitimate
/// per-frame absence is signalled by the *caller* (no matching object id), never by this
/// helper swallowing a shape error.
pub(crate) fn mask_to_frame(
    mask_logits: &[f32],
    grid: usize,
    width: u32,
    height: u32,
) -> WorkerResult<Vec<u8>> {
    if mask_logits.len() != grid * grid {
        return Err(crate::WorkerError::Engine(format!(
            "sam3 mask has {} logits, expected grid×grid = {}",
            mask_logits.len(),
            grid * grid
        )));
    }
    let bin: Vec<u8> = mask_logits
        .iter()
        .map(|&v| if v > 0.0 { 255 } else { 0 })
        .collect();
    let small = image::GrayImage::from_raw(grid as u32, grid as u32, bin).ok_or_else(|| {
        crate::WorkerError::Engine(format!(
            "sam3 mask buffer ({}) does not fit a {grid}×{grid} GrayImage",
            grid * grid
        ))
    })?;
    let resized =
        image::imageops::resize(&small, width, height, image::imageops::FilterType::Triangle);
    Ok(resized
        .into_raw()
        .into_iter()
        .map(|v| if v > 127 { 255 } else { 0 })
        .collect())
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

/// Emit the selected object's binary mask on one SAM3 frame, or an empty vec when the object isn't
/// present (legitimate per-frame absence → orchestrator box-fallback). Guards the `obj_ids`/`masks`
/// parallel-vec assumption: an id present in `obj_ids` but with no matching entry in `masks` is a
/// malformed engine output, surfaced as an `Engine` error rather than indexing OOB (sc-8905, F-103).
///
/// Generic over [`Sam3FrameOutput`] so the identical selection/emit runs on either backend's frame
/// type — the last verbatim duplicate between the two SAM3 twins (sc-11191, F-018).
pub(crate) fn frame_mask_for_object<F: Sam3FrameOutput>(
    frame: &F,
    selected: i32,
    width: u32,
    height: u32,
) -> WorkerResult<Vec<u8>> {
    let Some(i) = frame.obj_ids().iter().position(|&o| o == selected) else {
        return Ok(Vec::new());
    };
    let logits = frame.masks().get(i).ok_or_else(|| {
        WorkerError::Engine(format!(
            "sam3 frame has obj id {selected} at index {i} but only {} masks",
            frame.masks().len()
        ))
    })?;
    mask_to_frame(logits, MASK_GRID, width, height)
}

/// Stable left-to-right paint order for the SCAIL-2 painter: each object's centroid-x in the FIRST
/// frame it appears, ascending (tie-break on first-seen frame, then object id, so repeated runs
/// agree). Generic over [`Sam3FrameOutput`] so both SAM3 twins share ONE definition — a tie-break
/// tweak can no longer silently change SCAIL-2 palette assignment on one platform only (sc-11191,
/// F-018).
pub(crate) fn paint_order<F: Sam3FrameOutput>(outputs: &[F]) -> Vec<i32> {
    use std::collections::BTreeMap;
    let mut first_seen: BTreeMap<i32, (usize, f64)> = BTreeMap::new();
    for (f, frame) in outputs.iter().enumerate() {
        for (oid, logits) in frame.obj_ids().iter().zip(frame.masks()) {
            if first_seen.contains_key(oid) {
                continue;
            }
            if let Some(cx) = mask_centroid_x(logits, MASK_GRID) {
                first_seen.insert(*oid, (f, cx));
            }
        }
    }
    let mut order: Vec<i32> = first_seen.keys().copied().collect();
    order.sort_by(|a, b| {
        let (fa, xa) = first_seen[a];
        let (fb, xb) = first_seen[b];
        xa.partial_cmp(&xb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(fa.cmp(&fb))
            .then(a.cmp(b))
    });
    order
}

/// Per-frame `(obj_id, binary row-major `width*height` 0/255 mask)` lists — the body of
/// [`AllPersonMasks::per_frame`], one inner `Vec` per clip frame.
pub(crate) type PerFrameMasks = Vec<Vec<(i32, Vec<u8>)>>;

/// Map every SAM3 object present on each frame to its `(obj_id, binary mask)` at `width×height` — the
/// per-frame body of [`AllPersonMasks::per_frame`]. `zip` bounds to the shorter of obj_ids/masks (so
/// the parallel-vec assumption is safe here); a malformed (grid-mismatched) mask propagates as an
/// `Engine` error from [`mask_to_frame`] rather than the empty-vec sentinel (sc-8905, F-103). Generic
/// over [`Sam3FrameOutput`] so both SAM3 twins share ONE definition (sc-11191, F-018).
pub(crate) fn per_frame_masks<F: Sam3FrameOutput>(
    outputs: &[F],
    width: u32,
    height: u32,
) -> WorkerResult<PerFrameMasks> {
    outputs
        .iter()
        .map(|frame| {
            frame
                .obj_ids()
                .iter()
                .zip(frame.masks())
                .map(|(oid, logits)| Ok((*oid, mask_to_frame(logits, MASK_GRID, width, height)?)))
                .collect::<WorkerResult<Vec<_>>>()
        })
        .collect::<WorkerResult<Vec<_>>>()
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
    pub per_frame: PerFrameMasks,
    pub width: u32,
    pub height: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-9879 (F-077 follow-up): the first-party SAM3 weights repo is fetched at a pinned commit,
    /// never the mutable `main` branch, so a re-push (or a compromised token) can't silently swap the
    /// SAM3 checkpoint we load. Lock the constant to a real 40-hex lowercase commit id.
    #[test]
    fn seg_revision_is_pinned_commit_not_main() {
        assert_ne!(SEG_REVISION, "main", "SAM3 must pin a fixed revision");
        assert_eq!(
            SEG_REVISION.len(),
            40,
            "a pinned HF revision is a 40-char commit sha"
        );
        assert!(
            SEG_REVISION
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "the pinned revision must be lowercase hex"
        );
    }

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
        let out = mask_to_frame(&logits, grid, 8, 8).expect("well-formed grid resizes");
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|&v| v == 0 || v == 255), "output is binary");
        assert_eq!(out[0], 255, "top-left corner is foreground");
        assert_eq!(out[63], 0, "bottom-right corner is background");
    }

    /// sc-8905 / F-103: a logit buffer whose length doesn't match grid×grid returns an `Engine`
    /// error, not the empty-vec sentinel the orchestrator reads as "object absent → box
    /// fallback". Previously a malformed mask silently degraded to no-mask.
    #[test]
    fn mask_to_frame_rejects_mismatched_grid_length() {
        // grid says 4×4 (16 logits) but only 10 are supplied.
        let result = mask_to_frame(&[1.0f32; 10], 4, 8, 8);
        assert!(
            matches!(result, Err(crate::WorkerError::Engine(ref m)) if m.contains("logits")),
            "mismatched grid length must be rejected, got {result:?}"
        );
    }

    /// A `MASK_GRID²` mask with the top-left quadrant foreground, for the shared-helper tests below.
    fn quadrant_mask(top_left: bool) -> Vec<f32> {
        let g = MASK_GRID;
        let mut m = vec![-1.0f32; g * g];
        let (rows, cols) = if top_left {
            (0..g / 2, 0..g / 2)
        } else {
            (g / 2..g, g / 2..g)
        };
        for y in rows {
            for x in cols.clone() {
                m[y * g + x] = 1.0;
            }
        }
        m
    }

    /// sc-11191 / F-018 (was sc-8905 / F-103, per twin): the shared `frame_mask_for_object` guards
    /// the obj_ids/masks parallel-vec assumption — absent id → empty vec (legitimate box-fallback), a
    /// present-but-unmatched id → `Engine` error (not an OOB index), a well-formed pair → a binary
    /// mask at the requested dims. Exercised on the backend-neutral `TestFrame` so it covers the
    /// single copy both twins now delegate to.
    #[test]
    fn frame_mask_for_object_selects_guards_and_binarizes() {
        // obj id 5 present but `masks` empty → parallel-vec violation → Engine error.
        let malformed = TestFrame {
            obj_ids: vec![5],
            masks: vec![],
        };
        assert!(
            matches!(frame_mask_for_object(&malformed, 5, 8, 8), Err(crate::WorkerError::Engine(ref m)) if m.contains("masks")),
            "obj id without a mask must error"
        );

        // Selected object absent → empty vec (box fallback), not an error.
        let absent = TestFrame {
            obj_ids: vec![9],
            masks: vec![quadrant_mask(true)],
        };
        assert!(
            frame_mask_for_object(&absent, 5, 8, 8)
                .expect("absent is not an error")
                .is_empty(),
            "absent object → empty mask"
        );

        // Well-formed pair → binary mask at width*height.
        let present = TestFrame {
            obj_ids: vec![5],
            masks: vec![quadrant_mask(true)],
        };
        let mask = frame_mask_for_object(&present, 5, 8, 8).expect("well-formed");
        assert_eq!(mask.len(), 64);
        assert!(mask.iter().all(|&v| v == 0 || v == 255), "binary");
    }

    /// sc-11191 / F-018: the shared `paint_order` sorts objects by first-seen centroid-x ascending
    /// (leftmost first), tie-breaking on first-seen frame then object id — the SCAIL-2 palette order.
    /// A left object must precede a right one regardless of obj-id numbering.
    #[test]
    fn paint_order_sorts_left_to_right_by_first_seen_centroid() {
        // obj 8 sits left, obj 3 sits right — order must be by centroid (8 then 3), NOT by id.
        let outputs = vec![TestFrame {
            obj_ids: vec![8, 3],
            masks: vec![quadrant_mask(true), quadrant_mask(false)],
        }];
        assert_eq!(paint_order(&outputs), vec![8, 3]);

        // Empty-mask objects (no centroid) are dropped from the order.
        let empty = vec![TestFrame {
            obj_ids: vec![1],
            masks: vec![vec![-1.0f32; MASK_GRID * MASK_GRID]],
        }];
        assert!(paint_order(&empty).is_empty());
    }

    /// sc-11191 / F-018: the shared `per_frame_masks` maps every present object to `(obj_id, binary
    /// mask)` per frame, and propagates a malformed-mask `Engine` error rather than dropping it.
    #[test]
    fn per_frame_masks_maps_all_objects_and_propagates_errors() {
        let outputs = vec![
            TestFrame {
                obj_ids: vec![2, 4],
                masks: vec![quadrant_mask(true), quadrant_mask(false)],
            },
            TestFrame {
                obj_ids: vec![2],
                masks: vec![quadrant_mask(true)],
            },
        ];
        let per_frame = per_frame_masks(&outputs, 8, 8).expect("well-formed");
        assert_eq!(per_frame.len(), 2);
        assert_eq!(per_frame[0].len(), 2);
        assert_eq!(per_frame[0][0].0, 2);
        assert_eq!(per_frame[0][0].1.len(), 64);
        assert_eq!(per_frame[1].len(), 1);

        // A grid-mismatched mask surfaces as an Engine error (not a silently dropped frame).
        let bad = vec![TestFrame {
            obj_ids: vec![7],
            masks: vec![vec![1.0f32; 10]],
        }];
        assert!(matches!(
            per_frame_masks(&bad, 8, 8),
            Err(crate::WorkerError::Engine(_))
        ));
    }

    fn f011_test_settings(data_dir: PathBuf) -> Settings {
        Settings {
            api_url: "http://127.0.0.1:0".to_owned(),
            access_token: None,
            data_dir,
            config_dir: PathBuf::from("config"),
            worker_id: "test-worker".to_owned(),
            gpu_id: "cpu".to_owned(),
            is_child_worker: true,
            poll_seconds: 1,
            heartbeat_seconds: 5,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: "http://127.0.0.1:0".to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: 8u64 * 1024 * 1024 * 1024,
            max_model_url_bytes: 256u64 * 1024 * 1024 * 1024,
            allow_private_lora_urls: true,
            utility_workers: 1,
            backend_mlx_enabled: true,
            backend_candle_enabled: false,
            gpu_memory_limit_bytes: 0,
            external_model_roots: Vec::new(),
        }
    }

    /// sc-11175/F-011: `SCENEWORKS_SAM3_WEIGHTS` (a dir or its `model.safetensors`) must fail
    /// loudly when set-but-missing OR set to an incomplete dir (missing either file of the
    /// `{model.safetensors, tokenizer.json}` pair), instead of silently falling through to the
    /// cache/HF download. An unset pin (nothing staged) falls through to `Ok(None)`.
    ///
    /// Holds the crate-wide env lock across the whole staged mutation (set → assert → unset →
    /// assert), because `set_var` is process-global and the rest of the crate's tests run as threads
    /// in this same process. This used to claim "`RUST_TEST_THREADS=1` is forced workspace-wide, so
    /// the env mutation is serial" and take no lock — nothing sets that variable anywhere in the
    /// repo, so the tests were in fact fully parallel and the mutation was unsynchronized (sc-12380).
    #[test]
    fn resolve_segmenter_weights_env_pin_missing_and_incomplete_error_unset_falls_through() {
        let key = "SCENEWORKS_SAM3_WEIGHTS";
        // Holds the lock for the whole staged mutation AND restores the operator's prior value on
        // drop — including if an assertion below panics, which the hand-rolled restore tail could
        // not. The body re-points `key` freely inside the guard.
        let _env = crate::test_env::EnvVars::set(&[(key, "")]);
        let dir = tempfile::tempdir().expect("tempdir");
        let settings = f011_test_settings(dir.path().to_path_buf());

        // (a) pinned path does not exist at all → loud, via resolve_env_file_pin.
        std::env::set_var(key, dir.path().join("nonexistent-sam3-dir"));
        let missing = resolve_segmenter_weights(&settings);
        assert!(
            matches!(missing, Err(crate::WorkerError::InvalidPayload(ref m)) if m.contains(key) && m.contains("does not exist")),
            "a set-but-missing SAM3 pin must error loudly, got {missing:?}"
        );

        // (b) pinned dir exists but is missing tokenizer.json → still loud (the pair check).
        let incomplete = dir.path().join("sam3-incomplete");
        std::fs::create_dir_all(&incomplete).expect("mk dir");
        std::fs::write(incomplete.join(MODEL_FILE), b"weights").expect("write model");
        std::env::set_var(key, &incomplete);
        let partial = resolve_segmenter_weights(&settings);
        assert!(
            matches!(partial, Err(crate::WorkerError::InvalidPayload(ref m)) if m.contains(key) && m.contains(TOKENIZER_FILE)),
            "an incomplete SAM3 pin dir must error loudly, got {partial:?}"
        );

        // (c) unset + nothing staged → fall through.
        std::env::remove_var(key);
        let unset = resolve_segmenter_weights(&settings);
        assert!(
            matches!(unset, Ok(None)),
            "an unset SAM3 pin must fall through to Ok(None), got {unset:?}"
        );
        // `_env` restores the prior value on drop.
    }
}
