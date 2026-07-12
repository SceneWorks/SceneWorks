//! Native-MLX **SAM3** text-concept person segmentation on the Rust worker (epic 4910, sc-4926).
//!
//! The box-prompt-free upgrade of `person_segment` (SAM2). Instead of prompting the segmenter
//! with the selected person's ByteTrack box, this drives the SAM3 **Promptable Concept
//! Segmentation (PCS)** video pipeline (`mlx-gen-sam3` `Sam3VideoModel::propagate`) from the text
//! concept `"person"`: SAM3 detects *every* person on every frame and tracks them across the clip
//! with its own memory bank + identity bookkeeping, returning per-frame `obj_id → mask`.
//!
//! Replace-Person still needs *one* selected person's mask per frame, so the per-frame ByteTrack
//! box stops being a *prompt* and becomes an *association hint*: we pick the SAM3 object whose
//! masks best fall inside the selected track's boxes across the span, then emit that object's
//! per-frame mask. The downstream contract is identical to the SAM2 path — one binary `L` mask
//! per clip frame, written under `person-tracks/{track_id}/masks/` by the orchestrator in
//! `media_jobs::segment_assembly_frames` — so the replacement loader and Wan-VACE are unchanged.
//!
//! macOS-only, like `person_segment` / `person_jobs`: `mlx-gen-sam3` builds Apple MLX from source
//! and is meaningless off Apple Silicon. **Cross-platform divergence (surfaced, not silent — cf.
//! epic 3792):** the Python/torch SAM2 *box-prompt* path stays the Windows/Linux backend until a
//! parallel SAM3 backport; only the macOS MLX worker gets the text-concept upgrade today.
//!
//! Unlike SAM2 (converted `.pt` → MLX), SAM3 loads the **stock `facebook/sam3` checkpoint
//! directly** (`model.safetensors` + `tokenizer.json`); no conversion step. The model is
//! affine-quantized after load (`Sam3VideoModel::quantize`, sc-4925) — **Q8 by default**
//! (~0.9 GB, near-lossless), tunable via `SCENEWORKS_SAM3_QUANT` (`q8`/`q4`/`off`).

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::person_segment::{check_segment_canceled, SegmentProgress, CANCEL_MESSAGE};
use gen_core::CancelFlag;
use mlx_gen::weights::Weights;
use mlx_gen_sam3::{
    Sam3ImageSegmenter, Sam3TextConfig, Sam3Tokenizer, Sam3Tracker, Sam3VideoModel,
    VideoFrameOutput,
};
use mlx_rs::Array;

use crate::person_segment_sam3_common::{
    mask_centroid_x, mask_to_frame, normalize_chw, select_object, BoxNorm, Sam3FrameOutput,
    CONCEPT_PROMPT, INPUT_SIZE, MASK_GRID,
};
use crate::{WorkerError, WorkerResult};

// Backend-neutral SAM3 helpers now live in `person_segment_sam3_common` (sc-8847, F-045). Re-export
// the download helper + `AllPersonMasks` under this module's path so the existing callers
// (`media_jobs`, `video_jobs`, `segment_jobs`, `scail2_masks`) keep referencing
// `person_segment_sam3::…` unchanged.
pub(crate) use crate::person_segment_sam3_common::{ensure_segmenter_weights, AllPersonMasks};

/// Adapt this backend's `mlx-gen-sam3` `VideoFrameOutput` to the shared association math
/// ([`select_object`]); the two backends' `VideoFrameOutput` are the same shape but distinct
/// types, so each module impls the neutral accessor on its own.
impl Sam3FrameOutput for VideoFrameOutput {
    fn obj_ids(&self) -> &[i32] {
        &self.obj_ids
    }
    fn masks(&self) -> &[Vec<f32>] {
        &self.masks
    }
}

/// Affine-quantization bits for the segmenter, from `SCENEWORKS_SAM3_QUANT`: **Q8 by default**
/// (`8` — near-lossless, engine image Q8 IoU 0.9988, ~0.9 GB resident vs F32 ~3.2 GB), `q4` for
/// the smaller/lossier Q4, or `off`/`f32` to keep dense F32. `None` = no quantization.
fn quant_bits() -> Option<i32> {
    parse_quant_bits(&std::env::var("SCENEWORKS_SAM3_QUANT").unwrap_or_default())
}

/// Parse the `SCENEWORKS_SAM3_QUANT` value (split out so the mapping is unit-testable). Unset or
/// unrecognized → the safe Q8 default.
fn parse_quant_bits(value: &str) -> Option<i32> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "f32" | "none" | "0" => None,
        "q4" | "4" => Some(4),
        _ => Some(8),
    }
}

/// The parsed SAM3 checkpoint is cached process-wide (the 3.2 GB safetensors parse is the
/// expensive part). A **fresh** `Sam3VideoModel` is assembled from it per clip: the model carries
/// per-session tracking state (obj ids, memory banks) and exposes no reset, so reusing one across
/// clips would leak identities. Building from cached weights is cheap (layer assembly over
/// already-resident arrays). Mirrors the SAM2 predictor cache + poison-recovery idiom.
static WEIGHTS: OnceLock<Mutex<Option<Weights>>> = OnceLock::new();

/// Cache key for the smart-select single-image models: the resolved weight path + the quant tier in
/// effect. A change in either (different model snapshot, `SCENEWORKS_SAM3_QUANT` flip) rebuilds, so a
/// re-pinned/re-configured worker never serves a stale quantized instance.
type Sam3CacheKey = (PathBuf, Option<i32>);

thread_local! {
    /// Smart-select **box** path (sc-8846 / F-044): the quantized [`Sam3ImageSegmenter`] is cached
    /// per blocking thread, keyed by `(model_path, quant_bits)`, so repeated interactive clicks skip
    /// the per-call model build + `quantize(8)` (seconds of latency + a transient dense/quantized
    /// memory spike each click). Unlike the video paths this is **safe to reuse**: the image
    /// segmenter is logically stateless — `segment_with_boxes(&self, …)` runs a pure forward + mask
    /// post-process and retains no per-call image embedding / prompt / mask memory (the only interior
    /// mutability is a byte-identical position-embedding memo, not segmentation state). Thread-local
    /// (not a process-wide `Mutex`) because the model embeds an `Rc<Backbone>` → `!Send`/`!Sync`, so
    /// it cannot cross the `spawn_blocking` thread boundary in a shared `static`; this mirrors
    /// mlx-gen-sam3's own thread-local resize-matrix cache. One slot per key → interactive (serial)
    /// smart-select keeps a single quantized model (~0.9 GB Q8) resident, not a per-click copy.
    static BOX_SEGMENTER: RefCell<Option<(Sam3CacheKey, Sam3ImageSegmenter)>> =
        const { RefCell::new(None) };

    /// Smart-select **point** path (sc-8846 / F-044): the quantized [`Sam3Tracker`] cached per
    /// blocking thread, keyed by `(model_path, quant_bits)`. Same rationale as [`BOX_SEGMENTER`]:
    /// `segment_points(&self, …)` is a pure single-frame forward (encode frame → decode points) with
    /// no retained memory bank — the video memory bank is assembled in the propagate loop, never
    /// stored on the tracker — so reusing the instance across clicks is byte-identical.
    static POINT_TRACKER: RefCell<Option<(Sam3CacheKey, Sam3Tracker)>> = const { RefCell::new(None) };
}

/// A unit of work handed to the dedicated smart-select thread: a boxed closure that runs there and
/// signals completion through its own captured reply channel.
type Sam3Job = Box<dyn FnOnce() + Send + 'static>;

/// After this long with **no** smart-select activity the dedicated thread drops its resident
/// quantized models (~0.9 GB Q8 each) so an otherwise-idle worker doesn't pin them forever. This is
/// the deliberate equivalent of the pre-sc-11180 behavior, where a smart-select model lived in a
/// tokio **blocking-pool** thread-local and was freed only when tokio reaped that thread after its
/// idle keep-alive (~10 s). We keep a slightly longer window here (a stable dedicated thread never
/// gets reaped, and interactive clicks arrive in bursts with pauses) so a working session stays warm
/// while a user who has moved on gets the memory back. The next click rebuilds from the still-cached
/// dense [`WEIGHTS`] (layer assembly + `quantize` over already-resident arrays).
const SAM3_IDLE_DROP: std::time::Duration = std::time::Duration::from_secs(30);

/// Owns the single OS thread that every `!Send` smart-select SAM3 model is pinned to (sc-11180,
/// F-012).
///
/// The smart-select box/point paths run under `tokio::task::spawn_blocking` ([`segment_jobs`]), so
/// successive interactive clicks land on **arbitrary** tokio blocking-pool threads. With the
/// per-thread [`BOX_SEGMENTER`]/[`POINT_TRACKER`] caches (sc-8846), a click that hops to a fresh
/// blocking thread both rebuilds **and** retains its own ~0.9 GB Q8 model there until tokio reaps
/// that idle thread — so a burst of clicks interleaved with other blocking jobs could pin several GB
/// of **duplicate** model memory on a unified-memory Mac and re-pay the seconds-long build+quantize
/// the cache was meant to avoid.
///
/// Routing **both** smart-select operations through this one thread pins the `!Send` model
/// (`Rc<Backbone>`) to exactly one thread, so the thread-local caches above hold at most a single
/// instance each and it is built/quantized at most once. Requests serialize on the thread — which is
/// what per-thread caching already effectively forced (one model, one op at a time) — so no
/// throughput is lost. The dense checkpoint stays in the process-wide [`WEIGHTS`] cache, still shared
/// with the video paths that keep running on their own `spawn_blocking` threads.
struct Sam3Executor {
    tx: Mutex<std::sync::mpsc::Sender<Sam3Job>>,
}

static SAM3_EXECUTOR: OnceLock<Sam3Executor> = OnceLock::new();

/// The process-wide smart-select executor, spawned on first use.
fn sam3_executor() -> &'static Sam3Executor {
    SAM3_EXECUTOR.get_or_init(Sam3Executor::spawn)
}

impl Sam3Executor {
    fn spawn() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Sam3Job>();
        std::thread::Builder::new()
            .name("sam3-smart-select".into())
            .spawn(move || loop {
                match rx.recv_timeout(SAM3_IDLE_DROP) {
                    // Each job internally `catch_unwind`s (see `run_on_sam3_thread`), so a panicking
                    // model build/forward is surfaced to that one caller without unwinding here —
                    // the thread survives and keeps serving later clicks.
                    Ok(job) => job(),
                    // Idle window elapsed → free the resident quantized models (equivalent
                    // idle-drop). Cheap no-op when the caches are already empty.
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        BOX_SEGMENTER.with(|c| *c.borrow_mut() = None);
                        POINT_TRACKER.with(|c| *c.borrow_mut() = None);
                    }
                    // Every `Sender` dropped → the process is shutting down; let the thread exit
                    // rather than spin. (The `static` holds one `Sender` for the process lifetime, so
                    // in practice this only fires at teardown.)
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            })
            .expect("spawn sam3 smart-select executor thread");
        Self { tx: Mutex::new(tx) }
    }

    /// Enqueue a job. Returns an `Engine` error (never hangs) if the thread is somehow gone.
    fn submit(&self, job: Sam3Job) -> WorkerResult<()> {
        self.tx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .send(job)
            .map_err(|_| WorkerError::Engine("sam3 smart-select executor thread is gone".into()))
    }
}

/// Run `job` on the dedicated smart-select thread and block for its result, so the `!Send` SAM3
/// model it touches lives on exactly one thread (sc-11180, F-012). Each job is wrapped in
/// `catch_unwind` so a panic in the model build/forward is surfaced to **this** caller as an
/// `Engine` error instead of killing the shared thread (which would strand every later click); the
/// thread therefore survives job panics and only exits at worker shutdown. A dead thread or dropped
/// reply is reported as an error rather than hanging the caller.
fn run_on_sam3_thread<T: Send + 'static>(
    job: impl FnOnce() -> WorkerResult<T> + Send + 'static,
) -> WorkerResult<T> {
    let (reply_tx, reply_rx) = std::sync::mpsc::channel::<WorkerResult<T>>();
    let boxed: Sam3Job = Box::new(move || {
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).unwrap_or_else(|_| {
                Err(WorkerError::Engine(
                    "sam3 smart-select thread panicked".into(),
                ))
            });
        // The caller may have gone away (its task dropped); a closed reply channel is not an error.
        let _ = reply_tx.send(outcome);
    });
    sam3_executor().submit(boxed)?;
    reply_rx
        .recv()
        .map_err(|_| WorkerError::Engine("sam3 smart-select thread dropped the reply".into()))?
}

/// Load the shared dense checkpoint into the process-wide [`WEIGHTS`] cache (poison-recovery) and run
/// `build` against it — the common cache-miss body for the smart-select single-image builders.
fn with_cached_weights<T>(
    model_path: &Path,
    build: impl FnOnce(&Weights) -> WorkerResult<T>,
) -> WorkerResult<T> {
    let cell = WEIGHTS.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if guard.is_none() {
        let weights = Weights::from_file(model_path)
            .map_err(|e| WorkerError::Engine(format!("sam3 weights load: {e}")))?;
        *guard = Some(weights);
    }
    build(guard.as_ref().expect("weights loaded"))
}

/// Build a quantized `Sam3ImageSegmenter` from the shared (cached) dense checkpoint — the box
/// smart-select cache-miss body (sc-8846). Split out so the build + `quantize` cost is in one place.
fn build_box_segmenter(model_path: &Path, quant: Option<i32>) -> WorkerResult<Sam3ImageSegmenter> {
    with_cached_weights(model_path, |weights| {
        let mut model = Sam3ImageSegmenter::from_weights(weights)
            .map_err(|e| WorkerError::Engine(format!("sam3 image model build: {e}")))?;
        if let Some(bits) = quant {
            model
                .quantize(bits)
                .map_err(|e| WorkerError::Engine(format!("sam3 quantize q{bits}: {e}")))?;
        }
        Ok(model)
    })
}

/// Build a quantized `Sam3Tracker` from the shared (cached) dense checkpoint — the point
/// smart-select cache-miss body (sc-8846), sibling of [`build_box_segmenter`].
fn build_point_tracker(model_path: &Path, quant: Option<i32>) -> WorkerResult<Sam3Tracker> {
    with_cached_weights(model_path, |weights| {
        let mut tracker = Sam3Tracker::from_weights(weights)
            .map_err(|e| WorkerError::Engine(format!("sam3 tracker build: {e}")))?;
        if let Some(bits) = quant {
            tracker
                .quantize(bits)
                .map_err(|e| WorkerError::Engine(format!("sam3 tracker quantize q{bits}: {e}")))?;
        }
        Ok(tracker)
    })
}

/// Preprocess an RGB frame to the SAM3 input tensor: resize to a 1008×1008 square (bilinear,
/// matching the processor's fixed-square resize — *not* aspect-preserving), rescale to `[0,1]`,
/// normalize by mean/std `0.5` to `[-1,1]`, packed NCHW `[1,3,1008,1008]` f32.
fn input_tensor(img: &image::RgbImage) -> Array {
    let resized = image::imageops::resize(
        img,
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::Triangle,
    );
    let chw = normalize_chw(resized.as_raw(), INPUT_SIZE as usize);
    Array::from_slice(&chw, &[1, 3, INPUT_SIZE as i32, INPUT_SIZE as i32])
}

/// Segment the selected person across a clip with the native-MLX SAM3 **text-concept (PCS) video
/// pipeline** (sc-4926). `clip_frame_paths` is the contiguous detected span (clip-local frame `0`
/// = first detected frame); `anchors[i]` is the frame's ByteTrack box in normalized
/// `(x, y, width, height)` when frame `i` was detected, else `None`. At least one anchor must be
/// `Some` — it is the association hint, not a prompt.
///
/// SAM3 runs once over the whole span (`propagate("person")`), segmenting and tracking *all* people
/// with its own identities; we then [`select_object`] the id that best overlaps the anchors and
/// emit that object's per-frame mask (gap frames the detector missed are still covered when SAM3
/// tracked the person through them — the same "survives weak-detection frames" win the SAM2 video
/// predictor gave us, now without any box prompt).
///
/// Returns one binary mask (row-major `width*height`, `0`/`255`) per clip frame, in clip order; an
/// empty vec for a frame where the selected object was absent (orchestrator skips empties → box
/// fallback). The checkpoint parses once and is cached process-wide; run under `spawn_blocking`
/// (image decode + GPU inference are blocking).
///
/// `cancel` is the user-cancel flag (sc-8807): checked at the coarse phase boundaries here (frame
/// decode, the cold 3.2 GB checkpoint parse, model build + quantize) and threaded into the
/// engine's per-frame propagate cancel contract (gen-core d8038beb), so a tripped flag stops the
/// clip between frames with [`WorkerError::Canceled`]. `progress` is invoked
/// `(frame_index, total_frames)` after each propagated frame.
pub(crate) fn segment_track_blocking(
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    clip_frame_paths: Vec<PathBuf>,
    anchors: Vec<Option<BoxNorm>>,
    cancel: Option<CancelFlag>,
    mut progress: Option<SegmentProgress>,
) -> WorkerResult<Vec<Vec<u8>>> {
    // A frames/anchors length mismatch is a caller contract violation. The old `assert_eq!`
    // panicked inside `spawn_blocking`, which `media_jobs` absorbed as a silent "degraded"
    // (box-fallback) result rather than a surfaced error — return `InvalidPayload` so the
    // mismatch fails the job loudly (sc-8903, F-101). Kept in sync with the candle twin.
    if clip_frame_paths.len() != anchors.len() {
        return Err(WorkerError::InvalidPayload(format!(
            "segment clip frames ({}) and anchors ({}) length mismatch",
            clip_frame_paths.len(),
            anchors.len()
        )));
    }
    check_segment_canceled(cancel.as_ref())?;
    if !anchors.iter().any(Option::is_some) {
        return Err(WorkerError::InvalidPayload(
            "person segmentation clip needs at least one detected frame to associate".into(),
        ));
    }

    // Decode every clip frame to RGB8 (shared rendered size) and build the SAM3 input tensors.
    let mut frames: Vec<Array> = Vec::with_capacity(clip_frame_paths.len());
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
        frames.push(input_tensor(&img));
    }

    // Guard the cold 3.2 GB checkpoint parse + model build + quantize — the engine only observes
    // the flag once propagation starts.
    check_segment_canceled(cancel.as_ref())?;

    // Cached checkpoint; recover from a poisoned lock by dropping the cached weights and reloading
    // (mirrors person_segment / sc-4277 F-MLXW-13).
    let cell = WEIGHTS.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if guard.is_none() {
        let weights = Weights::from_file(&model_path)
            .map_err(|e| WorkerError::Engine(format!("sam3 weights load: {e}")))?;
        *guard = Some(weights);
    }
    let weights = guard.as_ref().expect("weights loaded");

    // Fresh model per clip (clean tracking state) + tokenize the concept once.
    let mut model = Sam3VideoModel::from_weights(weights)
        .map_err(|e| WorkerError::Engine(format!("sam3 model build: {e}")))?;
    // Quantize (Q8 default) for a ~0.9 GB footprint vs F32 ~3.2 GB (sc-4925). The dense path is
    // parity-preserving, so the F32 (`SCENEWORKS_SAM3_QUANT=off`) result is unchanged.
    if let Some(bits) = quant_bits() {
        model
            .quantize(bits)
            .map_err(|e| WorkerError::Engine(format!("sam3 quantize q{bits}: {e}")))?;
    }
    let tokenizer = Sam3Tokenizer::from_file(&tokenizer_path, &Sam3TextConfig::sam3())
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenizer load: {e}")))?;
    let (input_ids, text_mask) = tokenizer
        .encode(CONCEPT_PROMPT)
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenize: {e}")))?;
    // The quantize pass above is seconds-long on a cold start; re-check before committing to the
    // propagate loop.
    check_segment_canceled(cancel.as_ref())?;

    // gen-core d8038beb (sc-7176 pin sync): `propagate` takes `cancel` + per-frame `progress`
    // (the video per-step cancel contract). Thread the caller's flag/callback so a user cancel
    // stops between frames and each frame reports progress (sc-8807).
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
            mlx_gen::Error::Canceled => WorkerError::Canceled(CANCEL_MESSAGE.to_owned()),
            e => WorkerError::Engine(format!("sam3 propagate: {e}")),
        })?;

    // Associate SAM3's identities to the selected track, then emit that object's per-frame mask.
    let Some(selected) = select_object(&outputs, &anchors) else {
        // SAM3 found no "person" overlapping any anchor → no masks (degrade to box fallback).
        return Ok(vec![Vec::new(); clip_frame_paths.len()]);
    };
    let masks = outputs
        .iter()
        .map(|frame| frame_mask_for_object(frame, selected, width, height))
        .collect::<WorkerResult<Vec<_>>>()?;
    Ok(masks)
}

/// Emit the selected object's binary mask on one SAM3 frame, or an empty vec when the object
/// isn't present (legitimate per-frame absence → orchestrator box-fallback). Guards the
/// `obj_ids`/`masks` parallel-vec assumption: an id present in `obj_ids` but with no matching
/// entry in `masks` is a malformed engine output, surfaced as an `Engine` error rather than
/// indexing OOB (sc-8905, F-103).
fn frame_mask_for_object(
    frame: &VideoFrameOutput,
    selected: i32,
    width: u32,
    height: u32,
) -> WorkerResult<Vec<u8>> {
    let Some(i) = frame.obj_ids.iter().position(|&o| o == selected) else {
        return Ok(Vec::new());
    };
    let logits = frame.masks.get(i).ok_or_else(|| {
        WorkerError::Engine(format!(
            "sam3 frame has obj id {selected} at index {i} but only {} masks",
            frame.masks.len()
        ))
    })?;
    mask_to_frame(logits, MASK_GRID, width, height)
}

/// Normalize an `[x1, y1, x2, y2]` pixel box (clamped to the image) to SAM3's `[cx, cy, w, h]`
/// ∈ [0, 1]. SAM3 squashes the image to a fixed 1008² square (NOT aspect-preserving), so a box's
/// normalized source coordinates equal its normalized model-input coordinates — no letterbox math.
fn normalize_box_cxcywh(box_xyxy: [f32; 4], width: u32, height: u32) -> [f32; 4] {
    let (w, h) = (width.max(1) as f32, height.max(1) as f32);
    let x1 = box_xyxy[0].min(box_xyxy[2]).clamp(0.0, w);
    let y1 = box_xyxy[1].min(box_xyxy[3]).clamp(0.0, h);
    let x2 = box_xyxy[0].max(box_xyxy[2]).clamp(0.0, w);
    let y2 = box_xyxy[1].max(box_xyxy[3]).clamp(0.0, h);
    [
        ((x1 + x2) * 0.5 / w).clamp(0.0, 1.0),
        ((y1 + y2) * 0.5 / h).clamp(0.0, 1.0),
        ((x2 - x1) / w).clamp(0.0, 1.0),
        ((y2 - y1) / h).clamp(0.0, 1.0),
    ]
}

/// Smart-select BOX path (epic 6087, sc-6105): public entry point. Routes the whole compute onto the
/// ONE dedicated smart-select thread (sc-11180, F-012) so the `!Send` [`Sam3ImageSegmenter`] is
/// built/quantized/cached on exactly that thread — never duplicated across the tokio blocking-pool
/// threads `spawn_blocking` hops between. `concept` is copied to an owned `String` so the submitted
/// closure is `'static`; all other args are already owned/`Copy`. See [`segment_box_on_thread`] for
/// the segmentation itself.
pub(crate) fn segment_box_blocking(
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    image: image::RgbImage,
    box_xyxy: [f32; 4],
    concept: &str,
    threshold: f32,
    mask_threshold: f32,
) -> WorkerResult<Vec<u8>> {
    let concept = concept.to_owned();
    run_on_sam3_thread(move || {
        segment_box_on_thread(
            model_path,
            tokenizer_path,
            image,
            box_xyxy,
            &concept,
            threshold,
            mask_threshold,
        )
    })
}

/// Segment whatever lies under a single box prompt on ONE still image with the native-MLX SAM3
/// box-prompted PVS path ([`Sam3ImageSegmenter::segment_with_boxes`], epic 4910 sc-4923). `box_xyxy`
/// is in source-image pixel coords; `concept` is the optional text concept paired with the box
/// (empty = rely on the geometric prompt). Returns one binary mask (row-major `width*height`,
/// `0`/`255`, white = the selected region) at the source dims — the `maskAssetId` the editor's
/// inpaint flow (sc-2436/2476) consumes. Errors when SAM3 returns no instance for the box. Loads the
/// segmenter from the shared (cached) SAM3 checkpoint and quantizes it (Q8 default). **Runs on the
/// dedicated smart-select thread** (via [`segment_box_blocking`]); MLX is synchronous + holds the
/// autorelease pool.
fn segment_box_on_thread(
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    image: image::RgbImage,
    box_xyxy: [f32; 4],
    concept: &str,
    threshold: f32,
    mask_threshold: f32,
) -> WorkerResult<Vec<u8>> {
    let (width, height) = (image.width(), image.height());
    if width == 0 || height == 0 {
        return Err(WorkerError::InvalidPayload(
            "smart-select source image has zero dimension".into(),
        ));
    }
    let pixels = input_tensor(&image);

    let cxcywh = normalize_box_cxcywh(box_xyxy, width, height);
    let boxes = Array::from_slice(&cxcywh, &[1, 1, 4]);
    let box_labels = [1i32]; // a single positive box prompt

    // Cache the quantized segmenter per blocking thread, keyed by (weights, quant), so repeated
    // interactive clicks skip the per-call `from_weights` + `quantize` (sc-8846 / F-044). The
    // segmenter is stateless (`segment_with_boxes(&self, …)` is a pure forward + post-process), so
    // reuse is byte-identical; the dense checkpoint is still parsed once into the shared `WEIGHTS`.
    // The tokenizer is a cheap host-side parse — kept per-call (no GPU tensors, negligible cost).
    let quant = quant_bits();
    let key: Sam3CacheKey = (model_path.clone(), quant);
    let instances = BOX_SEGMENTER.with(|cell| -> WorkerResult<_> {
        {
            let cached = cell.borrow();
            if cached.as_ref().map(|(k, _)| k) != Some(&key) {
                drop(cached);
                let model = build_box_segmenter(&model_path, quant)?;
                *cell.borrow_mut() = Some((key.clone(), model));
            }
        }
        let cached = cell.borrow();
        let (_, model) = cached.as_ref().expect("segmenter cached");
        let tokenizer = Sam3Tokenizer::from_file(&tokenizer_path, &Sam3TextConfig::sam3())
            .map_err(|e| WorkerError::Engine(format!("sam3 tokenizer load: {e}")))?;
        let (input_ids, text_mask) = tokenizer
            .encode(concept)
            .map_err(|e| WorkerError::Engine(format!("sam3 tokenize: {e}")))?;
        model
            .segment_with_boxes(
                &pixels,
                &input_ids,
                &text_mask,
                &boxes,
                &box_labels,
                (width as f32, height as f32),
                threshold,
                mask_threshold,
            )
            .map_err(|e| WorkerError::Engine(format!("sam3 segment_with_boxes: {e}")))
    })?;

    // Pick the instance whose MASK has the most foreground inside the prompt box. SAM3 PVS returns
    // the box-echo query (its box ≈ the prompt, but the mask can be degenerate) alongside the
    // model's own detections (real masks, different boxes); selecting by box-IoU can land on the
    // empty echo. Mask-in-box intersection is robust for both a tight box (the echo's own real
    // mask) and a loose box (the best-overlapping real detection). Instance masks are 0/1 at the
    // 288² grid; the squashed grid maps directly to the normalized frame (uniform 1008² resize).
    let nx1 = (box_xyxy[0].min(box_xyxy[2]) / width as f32).clamp(0.0, 1.0);
    let ny1 = (box_xyxy[1].min(box_xyxy[3]) / height as f32).clamp(0.0, 1.0);
    let nx2 = (box_xyxy[0].max(box_xyxy[2]) / width as f32).clamp(0.0, 1.0);
    let ny2 = (box_xyxy[1].max(box_xyxy[3]) / height as f32).clamp(0.0, 1.0);
    let mut best: Option<(u64, usize, Vec<f32>)> = None;
    for inst in &instances {
        let grid = inst.mask.shape()[0] as usize;
        let m: Vec<f32> = inst
            .mask
            .as_dtype(mlx_rs::Dtype::Float32)
            .map_err(|e| WorkerError::Engine(format!("sam3 mask read: {e}")))?
            .as_slice::<f32>()
            .to_vec();
        let mut inside = 0u64;
        for gy in 0..grid {
            for gx in 0..grid {
                if m[gy * grid + gx] > 0.0 {
                    let cx = (gx as f32 + 0.5) / grid as f32;
                    let cy = (gy as f32 + 0.5) / grid as f32;
                    if cx >= nx1 && cx < nx2 && cy >= ny1 && cy < ny2 {
                        inside += 1;
                    }
                }
            }
        }
        if best.as_ref().map_or(true, |(b, _, _)| inside > *b) {
            best = Some((inside, grid, m));
        }
    }
    let (_, grid, grid_mask) = best.filter(|(inside, _, _)| *inside > 0).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "SAM3 found no object in the selection box — try a tighter box or use the brush."
                .into(),
        )
    })?;
    // Reuse the >0 binarize + resize-to-source path (inverts the 1008² squash back to the frame).
    mask_to_frame(&grid_mask, grid, width, height)
}

/// Smart-select POINT path (epic 6087, sc-6346): public entry point. Routes the whole compute onto
/// the ONE dedicated smart-select thread (sc-11180, F-012) so the `!Send` [`Sam3Tracker`] is
/// built/quantized/cached on exactly that thread — never duplicated across the tokio blocking-pool
/// threads `spawn_blocking` hops between. See [`segment_points_on_thread`] for the segmentation
/// itself.
pub(crate) fn segment_points_blocking(
    model_path: PathBuf,
    image: image::RgbImage,
    points: Vec<(f32, f32, i32)>,
) -> WorkerResult<Vec<u8>> {
    run_on_sam3_thread(move || segment_points_on_thread(model_path, image, points))
}

/// Segment whatever lies under fg/bg click points on ONE still image with the native-MLX SAM3
/// **tracker's** single-frame PVS point prompt ([`Sam3Tracker::segment_points`]). SAM3 does
/// interactive point refinement via its tracker (the SAM2-lineage promptable mask decoder); the box
/// smart-select (sc-6105) uses the concept *detector* (`Sam3ImageSegmenter::segment_with_boxes`)
/// while points use the *tracker* — but both load from the SAME `facebook/sam3` checkpoint, so no
/// second model/download. `points` are `(x, y, label)` in source-image pixel coords, `label` `1` =
/// foreground / `0` = background. Returns one binary mask (row-major `width*height`, `0`/`255`,
/// white = the selected region) at the source dims — the same `maskAssetId` shape the box path
/// returns for the editor's inpaint flow (sc-2436/2476). Loads the tracker from the shared (cached)
/// SAM3 checkpoint + quantizes it (Q8 default). **Runs on the dedicated smart-select thread** (via
/// [`segment_points_blocking`]); MLX is synchronous + holds the autorelease pool.
fn segment_points_on_thread(
    model_path: PathBuf,
    image: image::RgbImage,
    points: Vec<(f32, f32, i32)>,
) -> WorkerResult<Vec<u8>> {
    let (width, height) = (image.width(), image.height());
    if width == 0 || height == 0 {
        return Err(WorkerError::InvalidPayload(
            "smart-select source image has zero dimension".into(),
        ));
    }
    if points.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "smart-select point path needs at least one point".into(),
        ));
    }
    let pixels = input_tensor(&image);

    // SAM3 squashes the image to a fixed 1008² square (uniform, NOT aspect-preserving), so a source
    // point maps to model-input space by the per-axis scale 1008/W, 1008/H (mirrors the box path's
    // normalize_box_cxcywh — no letterbox math).
    let (sx, sy) = (
        INPUT_SIZE as f32 / width as f32,
        INPUT_SIZE as f32 / height as f32,
    );
    let coords: Vec<(f32, f32)> = points.iter().map(|&(x, y, _)| (x * sx, y * sy)).collect();
    let labels: Vec<i32> = points.iter().map(|&(_, _, l)| l).collect();

    // Cache the quantized tracker per blocking thread, keyed by (weights, quant), so repeated
    // interactive clicks skip the per-call `from_weights` + `quantize` (sc-8846 / F-044). The tracker
    // is stateless on this single-frame path (`segment_points(&self, …)` encodes the frame + decodes
    // the points, retaining no memory bank), so reuse is byte-identical; the dense checkpoint is
    // still parsed once into the shared `WEIGHTS`.
    let quant = quant_bits();
    let key: Sam3CacheKey = (model_path.clone(), quant);
    let mask = POINT_TRACKER.with(|cell| -> WorkerResult<_> {
        {
            let cached = cell.borrow();
            if cached.as_ref().map(|(k, _)| k) != Some(&key) {
                drop(cached);
                let tracker = build_point_tracker(&model_path, quant)?;
                *cell.borrow_mut() = Some((key.clone(), tracker));
            }
        }
        let cached = cell.borrow();
        let (_, tracker) = cached.as_ref().expect("tracker cached");
        tracker
            .segment_points(&pixels, &coords, &labels)
            .map_err(|e| WorkerError::Engine(format!("sam3 tracker segment_points: {e}")))
    })?;

    // Low-res mask logits `[mg, mg]` → binarize `> 0` + resize to source dims (inverts the squash).
    let grid = mask.low_res.shape()[0] as usize;
    let logits = mask
        .low_res
        .as_dtype(mlx_rs::Dtype::Float32)
        .map_err(|e| WorkerError::Engine(format!("sam3 mask read: {e}")))?
        .as_slice::<f32>()
        .to_vec();
    mask_to_frame(&logits, grid, width, height)
}

/// Segment + track every "person" across already-decoded RGB `frames` with the SAM3 text-concept
/// (PCS) video pipeline, returning all objects' per-frame masks + the left-to-right paint order.
/// In-memory sibling of [`segment_track_blocking`] (no temp frame files): the SCAIL-2 reference /
/// driving frames are already decoded `Image`s, so the temp-PNG round-trip is skipped. `frames` must
/// be non-empty and uniform-sized. The checkpoint parses once and is cached process-wide; run under
/// `spawn_blocking` (GPU inference is blocking).
///
/// `cancel`/`progress` follow [`segment_track_blocking`] (sc-8807): coarse checks around the cold
/// checkpoint parse + model build, the engine's per-frame cancel between frames, and a
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
    let tensors: Vec<Array> = frames.iter().map(input_tensor).collect();

    // Guard the cold 3.2 GB checkpoint parse + model build + quantize — the engine only observes
    // the flag once propagation starts.
    check_segment_canceled(cancel.as_ref())?;

    // Cached checkpoint; recover from a poisoned lock by dropping + reloading (mirrors
    // `segment_track_blocking` / sc-4277 F-MLXW-13).
    let cell = WEIGHTS.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if guard.is_none() {
        let weights = Weights::from_file(model_path)
            .map_err(|e| WorkerError::Engine(format!("sam3 weights load: {e}")))?;
        *guard = Some(weights);
    }
    let weights = guard.as_ref().expect("weights loaded");

    let mut model = Sam3VideoModel::from_weights(weights)
        .map_err(|e| WorkerError::Engine(format!("sam3 model build: {e}")))?;
    if let Some(bits) = quant_bits() {
        model
            .quantize(bits)
            .map_err(|e| WorkerError::Engine(format!("sam3 quantize q{bits}: {e}")))?;
    }
    let tokenizer = Sam3Tokenizer::from_file(tokenizer_path, &Sam3TextConfig::sam3())
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenizer load: {e}")))?;
    let (input_ids, text_mask) = tokenizer
        .encode(CONCEPT_PROMPT)
        .map_err(|e| WorkerError::Engine(format!("sam3 tokenize: {e}")))?;
    check_segment_canceled(cancel.as_ref())?;

    // gen-core d8038beb (sc-7176 pin sync): `propagate` takes `cancel` + per-frame `progress`
    // (the video per-step cancel contract). Thread the caller's flag/callback (sc-8807).
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
            mlx_gen::Error::Canceled => WorkerError::Canceled(CANCEL_MESSAGE.to_owned()),
            e => WorkerError::Engine(format!("sam3 propagate: {e}")),
        })?;

    // Paint order: each object's centroid-x in the FIRST frame it appears, ascending (tie-break on
    // first-seen frame, then object id, so repeated runs agree).
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

    // `zip` already bounds to the shorter of obj_ids/masks, so the parallel-vec assumption is
    // safe here; `mask_to_frame` now returns an `Engine` error on a malformed (grid-mismatched)
    // mask instead of the empty-vec sentinel, so propagate rather than silently dropping it
    // (sc-8905, F-103).
    let per_frame = outputs
        .iter()
        .map(|frame| {
            frame
                .obj_ids
                .iter()
                .zip(&frame.masks)
                .map(|(oid, logits)| Ok((*oid, mask_to_frame(logits, MASK_GRID, width, height)?)))
                .collect::<WorkerResult<Vec<_>>>()
        })
        .collect::<WorkerResult<Vec<_>>>()?;

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
    // The checkpoint filenames now live in the shared module; the real-weights smokes below join them
    // onto a snapshot dir to build the model/tokenizer paths.
    use crate::person_segment_sam3_common::{MODEL_FILE, TOKENIZER_FILE};

    /// sc-8846 / F-044: the smart-select model cache is keyed by `(model_path, quant_bits)`, so a
    /// re-pinned weight snapshot **or** a `SCENEWORKS_SAM3_QUANT` flip is a cache miss (rebuild),
    /// never a stale quantized instance served under a new configuration. Pins that discriminator
    /// without needing the 3.2 GB weights.
    #[test]
    fn cache_key_distinguishes_weights_and_quant_tier() {
        let a: Sam3CacheKey = (PathBuf::from("/models/sam3/model.safetensors"), Some(8));
        // Same key → cache hit.
        assert_eq!(
            a.clone(),
            (PathBuf::from("/models/sam3/model.safetensors"), Some(8))
        );
        // Different quant tier (Q8 → Q4) → miss.
        assert_ne!(
            a,
            (PathBuf::from("/models/sam3/model.safetensors"), Some(4))
        );
        // Quant off vs on → miss.
        assert_ne!(a, (PathBuf::from("/models/sam3/model.safetensors"), None));
        // Different weight snapshot → miss.
        assert_ne!(a, (PathBuf::from("/other/model.safetensors"), Some(8)));
    }

    /// sc-8807: a pre-tripped cancel flag short-circuits BEFORE frame decode / the cold 3.2 GB
    /// checkpoint parse — the paths point at nothing, so reaching either would error with
    /// `InvalidPayload`/`Engine` instead of `Canceled`. Pins the coarse cancel seam ordering on
    /// both video entry points.
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

    /// sc-8903 / F-101: a frames/anchors length mismatch returns `InvalidPayload` (a surfaced
    /// error) instead of the old `assert_eq!` panic that `media_jobs` absorbed as a silent
    /// "degraded" box-fallback. The length check runs before any frame decode / weight load, so
    /// the nonexistent paths are never touched.
    #[test]
    fn frames_anchors_length_mismatch_returns_invalid_payload() {
        let result = segment_track_blocking(
            PathBuf::from("/nonexistent/model.safetensors"),
            PathBuf::from("/nonexistent/tokenizer.json"),
            vec![
                PathBuf::from("/nonexistent/a.png"),
                PathBuf::from("/nonexistent/b.png"),
            ],
            vec![Some((0.1, 0.1, 0.5, 0.5))], // one anchor for two frames
            None,
            None,
        );
        assert!(
            matches!(result, Err(WorkerError::InvalidPayload(ref m)) if m.contains("length mismatch")),
            "expected InvalidPayload length mismatch, got {result:?}"
        );
    }

    /// sc-8905 / F-103: `frame_mask_for_object` guards the obj_ids/masks parallel-vec
    /// assumption. An id present in `obj_ids` but with no matching `masks` entry (a malformed
    /// engine output) surfaces an `Engine` error instead of indexing OOB; an absent id returns
    /// an empty vec (legitimate per-frame absence); a well-formed pair produces a mask.
    #[test]
    fn frame_mask_for_object_guards_parallel_vecs() {
        // obj id 5 is present but `masks` is empty → parallel-vec violation → Engine error.
        let malformed = VideoFrameOutput {
            obj_ids: vec![5],
            masks: vec![],
        };
        let err = frame_mask_for_object(&malformed, 5, 8, 8);
        assert!(
            matches!(err, Err(WorkerError::Engine(ref m)) if m.contains("masks")),
            "obj id without a mask must error, got {err:?}"
        );

        // Selected object absent from this frame → empty vec (box-fallback), not an error.
        let absent = VideoFrameOutput {
            obj_ids: vec![9],
            masks: vec![vec![-1.0f32; MASK_GRID * MASK_GRID]],
        };
        let empty = frame_mask_for_object(&absent, 5, 8, 8).expect("absent object is not an error");
        assert!(
            empty.is_empty(),
            "absent object → empty mask (box fallback)"
        );

        // Well-formed pair → a real binary mask at the requested dims.
        let present = VideoFrameOutput {
            obj_ids: vec![5],
            masks: vec![vec![1.0f32; MASK_GRID * MASK_GRID]],
        };
        let mask = frame_mask_for_object(&present, 5, 8, 8).expect("well-formed mask");
        assert_eq!(mask.len(), 64, "mask sized to width*height");
        assert!(mask.iter().all(|&v| v == 0 || v == 255), "mask is binary");
    }

    #[test]
    fn normalize_box_maps_xyxy_pixels_to_unit_cxcywh() {
        // A 100×200 box at (10,20) on a 200×400 image → center (60,120), size (100,200).
        let b = normalize_box_cxcywh([10.0, 20.0, 110.0, 220.0], 200, 400);
        assert!((b[0] - 0.30).abs() < 1e-6, "cx {}", b[0]); // 60/200
        assert!((b[1] - 0.30).abs() < 1e-6, "cy {}", b[1]); // 120/400
        assert!((b[2] - 0.50).abs() < 1e-6, "w {}", b[2]); // 100/200
        assert!((b[3] - 0.50).abs() < 1e-6, "h {}", b[3]); // 200/400
    }

    #[test]
    fn normalize_box_orders_corners_and_clamps_to_image() {
        // Reversed corners + out-of-bounds → ordered + clamped to [0,1].
        let b = normalize_box_cxcywh([300.0, -50.0, 50.0, 500.0], 200, 400);
        assert!(b.iter().all(|&v| (0.0..=1.0).contains(&v)), "{b:?}");
        // x spans [50,200] (clamped from 300) → cx = 125/200 = 0.625, w = 150/200 = 0.75
        assert!((b[0] - 0.625).abs() < 1e-6, "cx {}", b[0]);
        assert!((b[2] - 0.75).abs() < 1e-6, "w {}", b[2]);
    }

    /// sc-11180 / F-012: the dedicated smart-select executor runs EVERY submitted job on the SAME
    /// single OS thread — the property that pins the `!Send` SAM3 model to one thread so it is never
    /// duplicated across tokio blocking-pool threads. Submit a batch of jobs from several caller
    /// threads and assert they all report one stable executor thread id, distinct from every caller.
    #[test]
    fn executor_runs_all_jobs_on_one_stable_thread() {
        // A first job establishes the executor thread's id.
        let executor_tid =
            run_on_sam3_thread(|| Ok::<_, WorkerError>(std::thread::current().id())).unwrap();

        // Many more jobs, submitted from distinct caller threads, must all land on that same id.
        let mut callers = Vec::new();
        for _ in 0..8 {
            callers.push(std::thread::spawn(move || {
                run_on_sam3_thread(|| Ok::<_, WorkerError>(std::thread::current().id())).unwrap()
            }));
        }
        for c in callers {
            let tid = c.join().expect("caller thread");
            assert_eq!(
                tid, executor_tid,
                "every job must run on the one executor thread"
            );
        }
        // And it is not the calling test thread — the model genuinely lives elsewhere.
        assert_ne!(
            executor_tid,
            std::thread::current().id(),
            "executor must be a dedicated thread, not the caller"
        );
    }

    /// sc-11180 / F-012: results and errors propagate back from the executor thread unchanged.
    #[test]
    fn executor_propagates_results_and_errors() {
        let ok = run_on_sam3_thread(|| Ok::<_, WorkerError>(vec![1u8, 2, 3])).unwrap();
        assert_eq!(ok, vec![1, 2, 3], "value returned from the executor thread");

        let err =
            run_on_sam3_thread(|| Err::<Vec<u8>, _>(WorkerError::InvalidPayload("boom".into())));
        assert!(
            matches!(err, Err(WorkerError::InvalidPayload(ref m)) if m == "boom"),
            "error surfaced verbatim, got {err:?}"
        );
    }

    /// sc-11180 / F-012: a panicking job is caught and surfaced as an `Engine` error to that caller
    /// WITHOUT killing the shared thread — a subsequent job on the same executor still succeeds.
    #[test]
    fn executor_survives_a_panicking_job() {
        let panicked = run_on_sam3_thread(|| -> WorkerResult<()> {
            panic!("deliberate job panic");
        });
        assert!(
            matches!(panicked, Err(WorkerError::Engine(ref m)) if m.contains("panicked")),
            "panic surfaced as Engine error, got {panicked:?}"
        );
        // The thread survived: a later job runs normally on the same executor thread.
        let after = run_on_sam3_thread(|| Ok::<_, WorkerError>(42u32)).unwrap();
        assert_eq!(after, 42, "executor thread still serving after a job panic");
    }

    #[test]
    fn quant_bits_defaults_to_q8_and_honors_overrides() {
        assert_eq!(parse_quant_bits(""), Some(8), "unset → Q8 default");
        assert_eq!(parse_quant_bits("q8"), Some(8));
        assert_eq!(parse_quant_bits("8"), Some(8));
        assert_eq!(
            parse_quant_bits(" Q4 "),
            Some(4),
            "trimmed + case-insensitive"
        );
        assert_eq!(parse_quant_bits("4"), Some(4));
        assert_eq!(parse_quant_bits("off"), None);
        assert_eq!(parse_quant_bits("F32"), None);
        assert_eq!(parse_quant_bits("none"), None);
        assert_eq!(
            parse_quant_bits("garbage"),
            Some(8),
            "unrecognized → safe Q8"
        );
    }

    // The pure-helper tests (`normalize_chw`, `mask_box_containment`, `select_object`,
    // `mask_to_frame`) moved to `person_segment_sam3_common` (sc-8847, F-045) — they exercise the
    // backend-neutral math once, compiled on both the MLX and candle lanes.

    /// Real-weights integration smoke (sc-4926, H4): the whole SceneWorks-side pipe — preprocess →
    /// `Sam3VideoModel::propagate("person")` → associate to the anchor → emit a per-frame mask —
    /// against the stock `facebook/sam3` checkpoint. Proves the cutover actually segments a person,
    /// not just that the pure helpers work. `#[ignore]`d (needs the 3.2 GB weights + GPU); run with:
    ///   SCENEWORKS_SAM3_WEIGHTS=<facebook/sam3 snapshot dir> \
    ///   SCENEWORKS_SAM3_SMOKE_IMAGE=<person jpg/png, e.g. ultralytics zidane.jpg 1280×720> \
    ///   cargo test -p sceneworks-worker --release sam3_real_weights_person_smoke -- --ignored --nocapture
    #[test]
    #[ignore = "real SAM3 weights + GPU; set SCENEWORKS_SAM3_WEIGHTS + SCENEWORKS_SAM3_SMOKE_IMAGE"]
    fn sam3_real_weights_person_smoke() {
        let snap = std::env::var("SCENEWORKS_SAM3_WEIGHTS")
            .expect("set SCENEWORKS_SAM3_WEIGHTS to a facebook/sam3 snapshot dir");
        let dir = {
            let p = PathBuf::from(&snap);
            if p.is_file() {
                p.parent().unwrap().to_path_buf()
            } else {
                p
            }
        };
        let model = dir.join(MODEL_FILE);
        let tokenizer = dir.join(TOKENIZER_FILE);
        let image = PathBuf::from(
            std::env::var("SCENEWORKS_SAM3_SMOKE_IMAGE")
                .expect("set SCENEWORKS_SAM3_SMOKE_IMAGE to a person image"),
        );

        // Two-frame clip from the same still; an anchor over the prominent foreground person
        // (zidane.jpg: the central/right player). The mask must mostly fall inside the anchor.
        let anchor: BoxNorm = (0.40, 0.10, 0.55, 0.88);
        let masks = segment_track_blocking(
            model,
            tokenizer,
            vec![image.clone(), image],
            vec![Some(anchor), Some(anchor)],
            None,
            None,
        )
        .expect("segment_track_blocking");

        assert_eq!(masks.len(), 2, "one mask per clip frame");
        let (w, h) = (1280usize, 720usize);
        for (i, mask) in masks.iter().enumerate() {
            assert_eq!(mask.len(), w * h, "frame {i} mask size");
            let fg = mask.iter().filter(|&&v| v > 127).count();
            let frac = fg as f64 / (w * h) as f64;
            // A real person mask covers a non-trivial, non-whole region of the frame.
            assert!(
                (0.02..0.80).contains(&frac),
                "frame {i} foreground fraction {frac:.3} is implausible (empty or whole-frame)"
            );
            // Most of the emitted foreground sits inside the anchor box.
            let (bx, by, bw, bh) = anchor;
            let (x1, y1, x2, y2) = (bx, by, bx + bw, by + bh);
            let inside = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .filter(|&(x, y)| mask[y * w + x] > 127)
                .filter(|&(x, y)| {
                    let (cx, cy) = ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
                    cx >= x1 && cx < x2 && cy >= y1 && cy < y2
                })
                .count();
            let containment = inside as f64 / fg.max(1) as f64;
            assert!(
                containment > 0.5,
                "frame {i} mask containment in anchor {containment:.3} too low (wrong object?)"
            );
            eprintln!("frame {i}: fg_frac={frac:.3} containment={containment:.3}");
        }
    }

    /// Real-weights smoke for the smart-select box path (sc-6105): preprocess →
    /// `Sam3ImageSegmenter::segment_with_boxes` → pick-best-instance → mask at the source dims,
    /// against the stock `facebook/sam3` checkpoint. Proves a single box prompt segments the object
    /// under it (the backend of the sc-3751 canvas tool). `#[ignore]`d (needs the 3.2 GB weights +
    /// GPU); run with:
    ///   SCENEWORKS_SAM3_WEIGHTS=<facebook/sam3 snapshot dir> \
    ///   SCENEWORKS_SAM3_SMOKE_IMAGE=<jpg/png with a clear subject, e.g. zidane.jpg 1280×720> \
    ///   [SCENEWORKS_SAM3_SMOKE_CONCEPT=<optional text concept>] \
    ///   cargo test -p sceneworks-worker --release sam3_real_weights_box_segment_smoke -- --ignored --nocapture
    #[test]
    #[ignore = "real SAM3 weights + GPU; set SCENEWORKS_SAM3_WEIGHTS + SCENEWORKS_SAM3_SMOKE_IMAGE"]
    fn sam3_real_weights_box_segment_smoke() {
        let snap = std::env::var("SCENEWORKS_SAM3_WEIGHTS")
            .expect("set SCENEWORKS_SAM3_WEIGHTS to a facebook/sam3 snapshot dir");
        let dir = {
            let p = PathBuf::from(&snap);
            if p.is_file() {
                p.parent().unwrap().to_path_buf()
            } else {
                p
            }
        };
        let model = dir.join(MODEL_FILE);
        let tokenizer = dir.join(TOKENIZER_FILE);
        let image_path = PathBuf::from(
            std::env::var("SCENEWORKS_SAM3_SMOKE_IMAGE")
                .expect("set SCENEWORKS_SAM3_SMOKE_IMAGE to an image with a clear subject"),
        );
        let image = crate::image_decode::decode_image_any(&image_path)
            .expect("decode smoke image")
            .to_rgb8();
        let (w, h) = (image.width(), image.height());

        // A box around the prominent foreground subject (zidane.jpg: the central/right player), in
        // source pixels. The concept defaults to empty (geometric prompt only).
        let concept = std::env::var("SCENEWORKS_SAM3_SMOKE_CONCEPT").unwrap_or_default();
        // Default box tightly bounds zidane.jpg's right-hand player; override with
        // SCENEWORKS_SAM3_SMOKE_BOX="x1,y1,x2,y2" (source pixels) for a different image. A tight box
        // around ONE object is the realistic smart-select gesture (a loose box spanning multiple
        // objects is ambiguous by design).
        let box_xyxy = std::env::var("SCENEWORKS_SAM3_SMOKE_BOX")
            .ok()
            .and_then(|s| {
                let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                (v.len() == 4).then(|| [v[0], v[1], v[2], v[3]])
            })
            .unwrap_or([
                w as f32 * 0.581,
                h as f32 * 0.058,
                w as f32 * 0.894,
                h as f32 * 0.989,
            ]);
        let mask = segment_box_blocking(
            model.clone(),
            tokenizer.clone(),
            image.clone(),
            box_xyxy,
            &concept,
            0.5,
            0.5,
        )
        .expect("segment_box_blocking");

        // sc-8846 / F-044: a SECOND identical call must return the byte-identical mask. The first
        // call populated the thread-local quantized-segmenter cache; this run hits the cache. A drift
        // here would mean the cached instance carried per-call state across clicks (state bleed) —
        // the exact hazard the statelessness analysis ruled out. Same-thread (test runs inline), so
        // it exercises the warm-cache path directly.
        let mask2 = segment_box_blocking(model, tokenizer, image, box_xyxy, &concept, 0.5, 0.5)
            .expect("segment_box_blocking (cached)");
        assert_eq!(
            mask, mask2,
            "cached segmenter must reproduce the mask byte-for-byte (no state bleed)"
        );

        assert_eq!(mask.len(), (w * h) as usize, "mask size = source dims");
        assert!(mask.iter().all(|&v| v == 0 || v == 255), "mask is binary");
        let fg = mask.iter().filter(|&&v| v > 127).count();
        let frac = fg as f64 / (w * h) as f64;
        assert!(
            (0.02..0.90).contains(&frac),
            "foreground fraction {frac:.3} implausible (empty or whole-frame)"
        );
        // Most emitted foreground sits inside the prompt box.
        let (x1, y1, x2, y2) = (box_xyxy[0], box_xyxy[1], box_xyxy[2], box_xyxy[3]);
        let inside = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| mask[(y * w + x) as usize] > 127)
            .filter(|&(x, y)| {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                px >= x1 && px < x2 && py >= y1 && py < y2
            })
            .count();
        let containment = inside as f64 / fg.max(1) as f64;
        assert!(
            containment > 0.5,
            "mask containment in box {containment:.3} too low (wrong object?)"
        );
        eprintln!("box-segment smoke: fg_frac={frac:.3} containment={containment:.3} dims={w}x{h}");
    }

    /// Real-weights smoke for the smart-select POINT path (sc-6346): decode an image → a single
    /// foreground click on the subject → `segment_points_blocking` (SAM3 tracker single-frame PVS) →
    /// a binary mask at the source dims, against the stock `facebook/sam3` checkpoint. Proves a
    /// positive click segments the object under it (the engine half of the sc-3751 click tool).
    /// `#[ignore]`d (needs the 3.2 GB weights + GPU); run with:
    ///   SCENEWORKS_SAM3_WEIGHTS=<facebook/sam3 snapshot dir> \
    ///   SCENEWORKS_SAM3_SMOKE_IMAGE=<jpg/png with a clear subject, e.g. zidane.jpg 1280×720> \
    ///   [SCENEWORKS_SAM3_SMOKE_POINT="x,y"] \
    ///   cargo test -p sceneworks-worker --release sam3_real_weights_point_segment_smoke -- --ignored --nocapture
    #[test]
    #[ignore = "real SAM3 weights + GPU; set SCENEWORKS_SAM3_WEIGHTS + SCENEWORKS_SAM3_SMOKE_IMAGE"]
    fn sam3_real_weights_point_segment_smoke() {
        let snap = std::env::var("SCENEWORKS_SAM3_WEIGHTS")
            .expect("set SCENEWORKS_SAM3_WEIGHTS to a facebook/sam3 snapshot dir");
        let dir = {
            let p = PathBuf::from(&snap);
            if p.is_file() {
                p.parent().unwrap().to_path_buf()
            } else {
                p
            }
        };
        let model = dir.join(MODEL_FILE);
        let image_path = PathBuf::from(
            std::env::var("SCENEWORKS_SAM3_SMOKE_IMAGE")
                .expect("set SCENEWORKS_SAM3_SMOKE_IMAGE to an image with a clear subject"),
        );
        let image = crate::image_decode::decode_image_any(&image_path)
            .expect("decode smoke image")
            .to_rgb8();
        let (w, h) = (image.width(), image.height());

        // A single positive click on the prominent subject (default = image center; override with
        // SCENEWORKS_SAM3_SMOKE_POINT="x,y" in source pixels for a different image).
        let (px, py) = std::env::var("SCENEWORKS_SAM3_SMOKE_POINT")
            .ok()
            .and_then(|s| {
                let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                (v.len() == 2).then(|| (v[0], v[1]))
            })
            .unwrap_or((w as f32 * 0.5, h as f32 * 0.5));
        let mask = segment_points_blocking(model.clone(), image.clone(), vec![(px, py, 1)])
            .expect("segment_points_blocking");

        // sc-8846 / F-044: a SECOND identical call must reproduce the mask byte-for-byte — the warm
        // thread-local tracker cache carries no per-click state (statelessness verified).
        let mask2 = segment_points_blocking(model, image, vec![(px, py, 1)])
            .expect("segment_points_blocking (cached)");
        assert_eq!(
            mask, mask2,
            "cached tracker must reproduce the mask byte-for-byte (no state bleed)"
        );

        assert_eq!(mask.len(), (w * h) as usize, "mask size = source dims");
        assert!(mask.iter().all(|&v| v == 0 || v == 255), "mask is binary");
        let fg = mask.iter().filter(|&&v| v == 255).count();
        let frac = fg as f64 / (w * h) as f64;
        // A real subject mask covers a non-trivial, non-whole region.
        assert!(
            (0.001..0.95).contains(&frac),
            "foreground fraction {frac:.3} is implausible (empty or whole-frame)"
        );
        // A positive click must land inside its own selected region.
        let idx = (py as u32).min(h - 1) * w + (px as u32).min(w - 1);
        assert_eq!(
            mask[idx as usize], 255,
            "the clicked pixel must be foreground in the mask"
        );
        eprintln!("point-segment smoke: fg_frac={frac:.3} at click ({px},{py}) dims={w}x{h}");
    }
}
