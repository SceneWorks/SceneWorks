//! Off-Mac (Windows/CUDA) **SAM3** text-concept person segmentation on the Rust worker — the candle
//! sibling of [`crate::person_segment_sam3`] (epic 5482, sc-6247 under sc-5062).
//!
//! Drives `candle-gen-sam3`'s `Sam3VideoModel::propagate("person")` exactly as the macOS path drives
//! the MLX twin: SAM3 detects + tracks *every* person across the clip with its own memory bank, and
//! we pick the object whose masks best fall inside the selected ByteTrack track's boxes (the per-frame
//! box is an *association hint*, never a segmenter prompt) and emit that object's per-frame binary
//! mask. The downstream contract is identical to the Mac SAM3 / SAM2 paths — one `L` mask per clip
//! frame, written under `person-tracks/{track_id}/masks/` by `media_jobs::segment_assembly_frames` —
//! so this **replaces the off-Mac SAM2 box-prompt stub** (`maskState = "missing"`) with real masks.
//!
//! Off-Mac + `backend-candle` only (`candle-gen-sam3` builds candle/CUDA). It loads the **stock
//! `facebook/sam3` checkpoint directly** (`model.safetensors` + `tokenizer.json`; no conversion).
//! Affine quant (`Sam3VideoModel::quantize`, sc-6246) is available via `SCENEWORKS_SAM3_QUANT`
//! (`q8`/`q4`), but the off-Mac default is **dense** — quantizing SAM3's PE vision ViT backbone NaNs
//! (its massive activations overflow GGUF's f16 quant scale, sc-6361; NOT a candle/Blackwell bug —
//! candle GGUF quant is correct on sm_120), and dense is bit-exact and fits the box. The pure
//! association/mask helpers are shared line-for-line with the MLX module; only the seam is candle.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::default_device;
use candle_gen::gen_core::Quant;
use candle_gen_sam3::{Sam3TextConfig, Sam3Tokenizer, Sam3VideoModel, VideoFrameOutput, Weights};
use gen_core::CancelFlag;

use crate::person_segment_sam3_common::{
    mask_centroid_x, mask_to_frame, normalize_chw, select_object, BoxNorm, Sam3FrameOutput,
    CONCEPT_PROMPT, INPUT_SIZE, MASK_GRID,
};
use crate::{WorkerError, WorkerResult};

// Backend-neutral SAM3 helpers now live in `person_segment_sam3_common` (sc-8847, F-045). Re-export
// the download helper + `AllPersonMasks` under this module's path so the existing off-Mac callers
// (`media_jobs`, `video_jobs`, `scail2_masks`) keep referencing `person_segment_sam3_candle::…`
// unchanged.
pub(crate) use crate::person_segment_sam3_common::{ensure_segmenter_weights, AllPersonMasks};

/// Adapt this backend's `candle-gen-sam3` `VideoFrameOutput` to the shared association math
/// ([`select_object`]); the two backends' `VideoFrameOutput` are the same shape but distinct types,
/// so each module impls the neutral accessor on its own.
impl Sam3FrameOutput for VideoFrameOutput {
    fn obj_ids(&self) -> &[i32] {
        &self.obj_ids
    }
    fn masks(&self) -> &[Vec<f32>] {
        &self.masks
    }
}

/// Cancel copy surfaced when a segmentation is interrupted by a user cancel — the candle sibling
/// of `person_segment::CANCEL_MESSAGE` (Mac-only, so duplicated like the other shared helpers).
pub(crate) const CANCEL_MESSAGE: &str = "Person segmentation canceled by user.";

/// Per-propagated-frame progress callback `(frame_index, total_frames)` — the candle sibling of
/// `person_segment::SegmentProgress` (Mac-only, so duplicated like the other shared helpers).
pub(crate) type SegmentProgress = Box<dyn FnMut(usize, usize) + Send>;

/// Bail out with [`WorkerError::Canceled`] when the threaded flag has been tripped — the coarse
/// cancel seam guarding frame decode, the cold multi-GB checkpoint parse, and model build/quantize
/// (sc-8807). The same flag is also threaded into `Sam3VideoModel::propagate`'s per-frame cancel
/// contract (sc-8972, the candle sibling of gen-core d8038beb), so a tripped flag stops the clip
/// between propagated frames too.
fn check_segment_canceled(cancel: Option<&CancelFlag>) -> WorkerResult<()> {
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    Ok(())
}

/// Affine-quantization level for the segmenter, from `SCENEWORKS_SAM3_QUANT`: **dense by default**
/// off-Mac (quantizing SAM3's PE vision backbone NaNs — its massive activations overflow GGUF's f16
/// quant scale, sc-6361, not a candle/Blackwell bug); `q8`/`q4` opt back in. `None` = dense F32.
fn quant_level() -> Option<Quant> {
    parse_quant(&std::env::var("SCENEWORKS_SAM3_QUANT").unwrap_or_default())
}

/// Parse the `SCENEWORKS_SAM3_QUANT` value (split out so the mapping is unit-testable). The default
/// off-Mac is **dense** (`None`): quantizing SAM3's PE vision ViT backbone produces NaN masks — its
/// massive activations overflow GGUF's f16 q8_1 block scale (sc-6361), hardware-agnostic and NOT a
/// candle/Blackwell bug (candle GGUF quant is correct on sm_120; the heads quantize fine, only the
/// backbone breaks). Dense is bit-exact and ~3.4 GB fits the GPU-worker box, so quant buys ~nothing
/// for SAM3. `q8`/`q4` remain opt-in; `off`/`f32`/`none`/unset/unrecognized all stay dense.
fn parse_quant(value: &str) -> Option<Quant> {
    match value.trim().to_ascii_lowercase().as_str() {
        "q4" | "4" => Some(Quant::Q4),
        "q8" | "8" => Some(Quant::Q8),
        _ => None,
    }
}

/// The parsed SAM3 checkpoint is cached process-wide (the multi-GB safetensors parse is the expensive
/// part). A **fresh** `Sam3VideoModel` is assembled from it per clip: the model carries per-session
/// tracking state (obj ids, memory banks) with no reset, so reusing one across clips would leak
/// identities. Mirrors the MLX module's cache + poison-recovery idiom.
static WEIGHTS: OnceLock<Mutex<Option<Weights>>> = OnceLock::new();

/// Preprocess an RGB frame to the SAM3 input tensor: resize to a 1008×1008 square (bilinear, fixed-
/// square — *not* aspect-preserving), normalize by mean/std `0.5` to `[-1,1]`, packed NCHW
/// `[1,3,1008,1008]` f32 on `device`.
fn input_tensor(img: &image::RgbImage, device: &Device) -> WorkerResult<Tensor> {
    let resized = image::imageops::resize(
        img,
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::Triangle,
    );
    let chw = normalize_chw(resized.as_raw(), INPUT_SIZE as usize);
    let n = INPUT_SIZE as usize;
    Tensor::from_vec(chw, (1, 3, n, n), device)
        .map_err(|e| WorkerError::Engine(format!("sam3 input tensor: {e}")))
}

/// Segment the selected person across a clip with the off-Mac candle SAM3 **text-concept (PCS) video
/// pipeline** (sc-6247). `clip_frame_paths` is the contiguous detected span (clip-local frame `0` =
/// first detected frame); `anchors[i]` is the frame's ByteTrack box in normalized `(x, y, width,
/// height)` when frame `i` was detected, else `None`. At least one anchor must be `Some` — it is the
/// association hint, not a prompt.
///
/// Returns one binary mask (row-major `width*height`, `0`/`255`) per clip frame, in clip order; an
/// empty vec for a frame where the selected object was absent (orchestrator skips empties → box
/// fallback). The checkpoint parses once and is cached process-wide; run under `spawn_blocking` (image
/// decode + GPU inference are blocking).
///
/// `cancel` is the user-cancel flag (sc-8807): checked at the coarse phase boundaries here (frame
/// decode, cold checkpoint parse, model build/quantize) and threaded into the engine's per-frame
/// propagate cancel contract (sc-8972, the candle sibling of gen-core d8038beb), so a tripped flag
/// stops the clip between frames with [`WorkerError::Canceled`]. `progress` is invoked
/// `(frame_index, total_frames)` after each propagated frame.
pub(crate) fn segment_track_blocking(
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    clip_frame_paths: Vec<PathBuf>,
    anchors: Vec<Option<BoxNorm>>,
    cancel: Option<CancelFlag>,
    mut progress: Option<SegmentProgress>,
) -> WorkerResult<Vec<Vec<u8>>> {
    assert_eq!(
        clip_frame_paths.len(),
        anchors.len(),
        "frames/anchors mismatch"
    );
    check_segment_canceled(cancel.as_ref())?;
    if !anchors.iter().any(Option::is_some) {
        return Err(WorkerError::InvalidPayload(
            "person segmentation clip needs at least one detected frame to associate".into(),
        ));
    }

    let device = default_device().map_err(|e| WorkerError::Engine(format!("sam3 device: {e}")))?;

    // Decode every clip frame to RGB8 (shared rendered size) and build the SAM3 input tensors.
    let mut frames: Vec<Tensor> = Vec::with_capacity(clip_frame_paths.len());
    let (mut width, mut height) = (0u32, 0u32);
    for path in &clip_frame_paths {
        let img = crate::image_decode::decode_image_any(path)
            .map_err(|e| WorkerError::InvalidPayload(format!("person frame open: {e}")))?
            .to_rgb8();
        if width == 0 {
            (width, height) = (img.width(), img.height());
        } else if img.width() != width || img.height() != height {
            return Err(WorkerError::InvalidPayload(
                "person clip frames are not all the same size".into(),
            ));
        }
        frames.push(input_tensor(&img, &device)?);
    }

    // Guard the cold multi-GB checkpoint parse + model build — the engine only observes the flag
    // once propagation starts.
    check_segment_canceled(cancel.as_ref())?;

    // Cached checkpoint; recover from a poisoned lock by dropping the cached weights and reloading.
    let cell = WEIGHTS.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if guard.is_none() {
        let weights = Weights::from_file(&model_path, &device)
            .map_err(|e| WorkerError::Engine(format!("sam3 weights load: {e}")))?;
        *guard = Some(weights);
    }
    let weights = guard.as_ref().expect("weights loaded");

    // Fresh model per clip (clean tracking state) + tokenize the concept once.
    let mut model = Sam3VideoModel::from_weights(weights)
        .map_err(|e| WorkerError::Engine(format!("sam3 model build: {e}")))?;
    // Optional quant (opt-in via `SCENEWORKS_SAM3_QUANT`); off-Mac defaults to dense (sc-6361), so
    // unset leaves the F32 result unchanged.
    if let Some(quant) = quant_level() {
        model
            .quantize(quant)
            .map_err(|e| WorkerError::Engine(format!("sam3 quantize: {e}")))?;
    }
    let tokenizer = Sam3Tokenizer::from_file(&tokenizer_path, &Sam3TextConfig::sam3())
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenizer load: {e}")))?;
    let (input_ids, text_mask) = tokenizer
        .encode(CONCEPT_PROMPT, &device)
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenize: {e}")))?;
    check_segment_canceled(cancel.as_ref())?;

    // candle-gen-sam3 `propagate` takes `cancel` + per-frame `progress` (sc-8972, the candle
    // sibling of the MLX video per-step cancel contract, gen-core d8038beb). Thread the caller's
    // flag/callback so a user cancel stops between frames and each frame reports progress.
    let outputs = model
        .propagate(
            &frames,
            &input_ids,
            &text_mask,
            cancel.as_ref(),
            progress
                .as_deref_mut()
                .map(|cb| cb as &mut dyn FnMut(usize, usize)),
        )
        .map_err(|e| match e {
            candle_gen::CandleError::Canceled => WorkerError::Canceled(CANCEL_MESSAGE.to_owned()),
            e => WorkerError::Engine(format!("sam3 propagate: {e}")),
        })?;

    // Associate SAM3's identities to the selected track, then emit that object's per-frame mask.
    let Some(selected) = select_object(&outputs, &anchors) else {
        // SAM3 found no "person" overlapping any anchor → no masks (degrade to box fallback).
        return Ok(vec![Vec::new(); clip_frame_paths.len()]);
    };
    let masks = outputs
        .iter()
        .map(|frame| {
            frame
                .obj_ids
                .iter()
                .position(|&o| o == selected)
                .map(|i| mask_to_frame(&frame.masks[i], MASK_GRID, width, height))
                .unwrap_or_default()
        })
        .collect();
    Ok(masks)
}

/// Segment + track every "person" across already-decoded RGB `frames` with the off-Mac candle SAM3
/// text-concept (PCS) video pipeline (sc-6837), returning all objects' per-frame masks + the
/// left-to-right paint order — the input to [`crate::scail2_masks`]. The candle sibling of
/// `crate::person_segment_sam3::segment_all_persons_in_memory`; in-memory (no temp frame files, unlike
/// [`segment_track_blocking`]) since the SCAIL-2 reference / driving frames are already decoded
/// `Image`s. `frames` must be non-empty and uniform-sized. The checkpoint parses once and is cached
/// process-wide; a fresh `Sam3VideoModel` is built per call (clean tracking state). Run under
/// `spawn_blocking` (GPU inference is blocking).
///
/// `cancel`/`progress` follow [`segment_track_blocking`] (sc-8807): coarse checks around the cold
/// checkpoint parse + model build, the engine's per-frame cancel between frames (sc-8972), and a
/// `(frame_index, total_frames)` callback after each propagated frame.
pub(crate) fn segment_all_persons_in_memory(
    model_path: &Path,
    tokenizer_path: &Path,
    frames: &[image::RgbImage],
    cancel: Option<CancelFlag>,
    mut progress: Option<SegmentProgress>,
) -> WorkerResult<AllPersonMasks> {
    check_segment_canceled(cancel.as_ref())?;
    let first = frames.first().ok_or_else(|| {
        WorkerError::InvalidPayload("scail2 segmentation: no frames to segment".into())
    })?;
    let (width, height) = (first.width(), first.height());
    if frames
        .iter()
        .any(|f| f.width() != width || f.height() != height)
    {
        return Err(WorkerError::InvalidPayload(
            "scail2 segmentation: frames are not all the same size".into(),
        ));
    }

    let device = default_device().map_err(|e| WorkerError::Engine(format!("sam3 device: {e}")))?;
    let tensors: Vec<Tensor> = frames
        .iter()
        .map(|f| input_tensor(f, &device))
        .collect::<WorkerResult<Vec<_>>>()?;

    // Guard the cold multi-GB checkpoint parse + model build — the engine only observes the flag
    // once propagation starts.
    check_segment_canceled(cancel.as_ref())?;

    // Cached checkpoint; recover from a poisoned lock by dropping + reloading (mirrors
    // `segment_track_blocking`).
    let cell = WEIGHTS.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if guard.is_none() {
        let weights = Weights::from_file(model_path, &device)
            .map_err(|e| WorkerError::Engine(format!("sam3 weights load: {e}")))?;
        *guard = Some(weights);
    }
    let weights = guard.as_ref().expect("weights loaded");

    // Fresh model per call (clean tracking state) + tokenize the concept once. Off-Mac defaults to
    // dense (sc-6361); `SCENEWORKS_SAM3_QUANT` opts back in.
    let mut model = Sam3VideoModel::from_weights(weights)
        .map_err(|e| WorkerError::Engine(format!("sam3 model build: {e}")))?;
    if let Some(quant) = quant_level() {
        model
            .quantize(quant)
            .map_err(|e| WorkerError::Engine(format!("sam3 quantize: {e}")))?;
    }
    let tokenizer = Sam3Tokenizer::from_file(tokenizer_path, &Sam3TextConfig::sam3())
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenizer load: {e}")))?;
    let (input_ids, text_mask) = tokenizer
        .encode(CONCEPT_PROMPT, &device)
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenize: {e}")))?;
    check_segment_canceled(cancel.as_ref())?;

    // candle-gen-sam3 `propagate` takes `cancel` + per-frame `progress` (sc-8972). Thread the
    // caller's flag/callback (mirrors `segment_track_blocking`).
    let outputs = model
        .propagate(
            &tensors,
            &input_ids,
            &text_mask,
            cancel.as_ref(),
            progress
                .as_deref_mut()
                .map(|cb| cb as &mut dyn FnMut(usize, usize)),
        )
        .map_err(|e| match e {
            candle_gen::CandleError::Canceled => WorkerError::Canceled(CANCEL_MESSAGE.to_owned()),
            e => WorkerError::Engine(format!("sam3 propagate: {e}")),
        })?;

    // Paint order: each object's centroid-x in the FIRST frame it appears, ascending (tie-break on
    // first-seen frame, then object id, so repeated runs agree). Shared with the MLX module.
    use std::collections::BTreeMap;
    let mut first_seen: BTreeMap<i32, (usize, f64)> = BTreeMap::new();
    for (f, frame) in outputs.iter().enumerate() {
        for (oid, logits) in frame.obj_ids.iter().zip(&frame.masks) {
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

    let per_frame = outputs
        .iter()
        .map(|frame| {
            frame
                .obj_ids
                .iter()
                .zip(&frame.masks)
                .map(|(oid, logits)| (*oid, mask_to_frame(logits, MASK_GRID, width, height)))
                .filter(|(_, mask)| !mask.is_empty())
                .collect()
        })
        .collect();

    Ok(AllPersonMasks {
        order,
        per_frame,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_defaults_to_dense_and_honors_opt_in() {
        assert_eq!(
            parse_quant(""),
            None,
            "unset → dense (SAM3 backbone quant overflows GGUF f16 scale, sc-6361)"
        );
        assert_eq!(parse_quant("q8"), Some(Quant::Q8), "explicit opt-in");
        assert_eq!(parse_quant("8"), Some(Quant::Q8));
        assert_eq!(
            parse_quant(" Q4 "),
            Some(Quant::Q4),
            "trimmed + case-insensitive"
        );
        assert_eq!(parse_quant("4"), Some(Quant::Q4));
        assert_eq!(parse_quant("off"), None);
        assert_eq!(parse_quant("F32"), None);
        assert_eq!(parse_quant("none"), None);
        assert_eq!(parse_quant("garbage"), None, "unrecognized → dense");
    }

    /// sc-8807: a pre-tripped cancel flag short-circuits BEFORE frame decode / the cold multi-GB
    /// checkpoint parse — the paths point at nothing, so reaching either would error with
    /// `InvalidPayload`/`Engine` instead of `Canceled`. Pins the coarse cancel seam ordering.
    #[test]
    fn pre_tripped_cancel_short_circuits_both_video_paths() {
        let cancel = CancelFlag::new();
        cancel.cancel();
        let track = segment_track_blocking(
            PathBuf::from("/nonexistent/model.safetensors"),
            PathBuf::from("/nonexistent/tokenizer.json"),
            vec![PathBuf::from("/nonexistent/frame.png")],
            vec![Some((0.1, 0.1, 0.5, 0.5))],
            Some(cancel.clone()),
            None,
        );
        assert!(
            matches!(track, Err(WorkerError::Canceled(_))),
            "segment_track_blocking: expected Canceled, got {track:?}"
        );
        let frames = vec![image::RgbImage::new(4, 4)];
        let all = segment_all_persons_in_memory(
            Path::new("/nonexistent/model.safetensors"),
            Path::new("/nonexistent/tokenizer.json"),
            &frames,
            Some(cancel),
            None,
        );
        assert!(
            matches!(all, Err(WorkerError::Canceled(_))),
            "segment_all_persons_in_memory: expected Canceled"
        );
    }

    // The pure-helper tests (`normalize_chw`, `mask_box_containment`, `select_object`,
    // `mask_to_frame`) moved to `person_segment_sam3_common` (sc-8847, F-045) — they exercise the
    // backend-neutral math once, compiled on both the MLX and candle lanes.
}
