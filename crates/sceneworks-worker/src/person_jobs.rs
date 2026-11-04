//! Person-detection orchestration, model resolution and backend-neutral JSON.
//!
//! The pinned runtime's CV engine owns YOLO11m execution, letterboxing, decode and
//! NMS. macOS uses the native MLX fused-weight network; the off-Mac candle lane
//! uses the canonical ONNX export with CUDA and CPU fallback. Both return original
//! frame pixel boxes for COCO class 0. The worker preserves its process cache,
//! instrumented weight loading, frame decoding and normalized result contract.
//!
//! CoreML is deliberately not used for YOLO11: the existing ONNX export hangs in
//! CoreML graph compilation. DWPose and Real-ESRGAN retain CoreML on macOS.
//! Person segmentation remains separately orchestrated through SAM2/SAM3.
//!
//! Existing ignored oracle/provider tests are explicitly dispatched by
//! `cv-provider-validation.yml`; ordinary fixtures live with the inference engine.

#[cfg(target_os = "macos")]
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::cv::person::Detection;
#[cfg(not(target_os = "macos"))]
use crate::cv::person::OrtYolo;
#[cfg(target_os = "macos")]
use crate::cv::person::Yolo;
use serde_json::{json, Value};

use crate::{Settings, WorkerError, WorkerResult};

/// The detector weights in the app cache / model dir: the fused MLX safetensors on macOS, the
/// canonical ONNX export off-Mac (the `ort` lane, sc-5498).
#[cfg(target_os = "macos")]
const DET_FILE: &str = "yolo11m_fused_mlx.safetensors";
/// Hugging Face repo for the detector weights (MLX safetensors on macOS / ONNX off-Mac).
#[cfg(target_os = "macos")]
const DET_REPO: &str = "SceneWorks/yolo11m-person-detect-mlx";
#[cfg(not(target_os = "macos"))]
const DET_FILE: &str = "yolo11m.onnx";
#[cfg(not(target_os = "macos"))]
const DET_REPO: &str = "SceneWorks/yolo11m-person-detect-onnx";
/// Pinned detector-weights revision (sc-9682, F-077 follow-up). Even though `DET_REPO` is
/// a first-party repo, fetching the mutable `main` branch means a re-push (or a compromised
/// token) could silently swap the detector weights we load. Pin the exact commit per repo
/// for defense-in-depth, mirroring the SeedVR2 pin (sc-8879). HF's tree API still reports
/// each file's `lfs.oid`, which `ensure_hf_cached_file` verifies the content against.
#[cfg(target_os = "macos")]
pub(crate) const DET_REVISION: &str = "d7027d3a8812bdebbf7862fc1c7dcfdeebb0f777";
#[cfg(not(target_os = "macos"))]
pub(crate) const DET_REVISION: &str = "3cffbaccca4f239ae6301d8b66ba721401ecfa8d";
// ---------------------------------------------------------------------------
// pure detector math (unit-tested without weights)
// ---------------------------------------------------------------------------

/// Convert NMS'd person boxes (original px) into the Python `run_person_detect`
/// `detections` array shape: `{id,label,box{x,y,width,height},confidence,
/// frameWidth,frameHeight,maskState}`, sorted confidence-descending, dropping
/// degenerate boxes. Mirrors `PersonDetection.to_dict` + `xyxy_to_normalized`.
pub(crate) fn detections_to_json(dets: &[Detection], frame_w: u32, frame_h: u32) -> Vec<Value> {
    if frame_w == 0 || frame_h == 0 {
        return Vec::new();
    }
    // Normalize in f64 to mirror Python `xyxy_to_normalized` (and avoid f32→f64
    // widening artifacts in the emitted JSON numbers).
    let (fw, fh) = (frame_w as f64, frame_h as f64);
    let mut sorted = dets.to_vec();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = Vec::new();
    for det in sorted {
        let x = (det.x1 as f64 / fw).clamp(0.0, 1.0);
        let y = (det.y1 as f64 / fh).clamp(0.0, 1.0);
        let width = (det.width() as f64 / fw).clamp(0.0, 1.0);
        let height = (det.height() as f64 / fh).clamp(0.0, 1.0);
        if width <= 0.0 || height <= 0.0 {
            continue;
        }
        let index = out.len() + 1;
        out.push(json!({
            "id": format!("person_{index}"),
            "label": format!("Person {index}"),
            "box": { "x": x, "y": y, "width": width, "height": height },
            "confidence": (det.score as f64 * 10000.0).round() / 10000.0,
            "frameWidth": frame_w,
            "frameHeight": frame_h,
            "maskState": "missing",
        }));
    }
    out
}

#[cfg(target_os = "macos")]
fn load_yolo(path: &Path) -> WorkerResult<Yolo> {
    let weights = crate::image_jobs::read_weights(path)
        .map_err(|error| WorkerError::Engine(format!("yolo11m weights load: {error}")))?;
    Ok(Yolo::from_weights(weights))
}

/// Person detections for one frame, plus the device the model ran on. Backend-neutral —
/// the macOS MLX path and the off-Mac `ort` path both return this.
pub(crate) struct DetectResult {
    pub width: u32,
    pub height: u32,
    pub detections: Vec<Detection>,
    pub device: &'static str,
}

#[cfg(target_os = "macos")]
static DETECTOR: OnceLock<Mutex<Option<Yolo>>> = OnceLock::new();

/// Blocking person detection on a single rendered frame (macOS native-MLX backend). The
/// model is loaded once and cached process-wide (like Python's lazy model load); invoke
/// via `spawn_blocking`.
#[cfg(target_os = "macos")]
pub(crate) fn detect_people_blocking(
    weights_path: PathBuf,
    image_path: PathBuf,
    conf: f32,
) -> WorkerResult<DetectResult> {
    let img = crate::image_decode::decode_image_any(&image_path)
        .map_err(|e| WorkerError::InvalidPayload(format!("person frame open: {e}")))?
        .to_rgb8();
    let (width, height) = (img.width(), img.height());

    let cell = DETECTOR.get_or_init(|| Mutex::new(None));
    // Recover from a poisoned lock instead of panicking every subsequent job: if a
    // prior detection panicked mid-run holding this lock, take the inner guard and
    // drop the possibly-corrupt cached model so the block below reloads a fresh one
    // (sc-4277 / F-MLXW-13).
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if guard.is_none() {
        *guard = Some(load_yolo(&weights_path)?);
    }
    let detector = guard.as_ref().expect("detector loaded");
    let detections = detector.detect_people(&img, conf)?;
    Ok(DetectResult {
        width,
        height,
        detections,
        device: "mlx",
    })
}

// ---------------------------------------------------------------------------
// off-Mac onnxruntime YOLO11 detector (sc-5498) — the Windows/Linux candle-lane sibling
// of the native-MLX forward above. Runs the canonical `yolo11m.onnx` export through `ort`
// with the CUDA EP (+ CPU fallback) and feeds its `(1,84,8400)` output into the shared
// `decode`/`nms` math, exactly as the MLX path does.
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
static DETECTOR: OnceLock<Mutex<Option<(PathBuf, OrtYolo)>>> = OnceLock::new();

/// Blocking person detection on a single rendered frame (off-Mac `ort`/CUDA backend). The
/// session is loaded once and cached process-wide; invoke via `spawn_blocking`.
#[cfg(not(target_os = "macos"))]
pub(crate) fn detect_people_blocking(
    weights_path: PathBuf,
    image_path: PathBuf,
    conf: f32,
) -> WorkerResult<DetectResult> {
    let img = crate::image_decode::decode_image_any(&image_path)
        .map_err(|e| WorkerError::InvalidPayload(format!("person frame open: {e}")))?
        .to_rgb8();
    let (width, height) = (img.width(), img.height());

    let cell = DETECTOR.get_or_init(|| Mutex::new(None));
    // Recover from a poisoned lock instead of panicking every subsequent job (mirrors the
    // macOS path): drop the possibly-corrupt cached session so the block below reloads.
    let mut guard = cell.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        guard
    });
    if !guard.as_ref().is_some_and(|(cached_path, _)| {
        crate::util::resolved_cache_paths_match([cached_path.as_path()], [weights_path.as_path()])
    }) {
        *guard = Some((
            weights_path.clone(),
            OrtYolo::load(&weights_path, &mut crate::cv::ModelIoLoader)?,
        ));
    }
    let detector = &mut guard.as_mut().expect("detector loaded").1;
    let device = detector.device;
    let detections = detector.detect_people(&img, conf)?;
    Ok(DetectResult {
        width,
        height,
        detections,
        device,
    })
}

// ---------------------------------------------------------------------------
// weights resolution (backend-neutral: the macOS MLX safetensors
// or the off-Mac ONNX export, selected by the cfg-split `DET_FILE`/`DET_REPO`)
// ---------------------------------------------------------------------------

/// Resolve already-present detector weights (the MLX safetensors on macOS / the ONNX export
/// off-Mac, per `DET_FILE`): explicit env pin (`SCENEWORKS_PERSON_DETECTOR_WEIGHTS`), then the
/// app cache `<data_dir>/cache/person-detect/`, then the model dir
/// `<data_dir>/models/person-detect/`. Returns `Ok(None)` when nothing is staged (then
/// `require_detector_weights` turns that into an actionable install error).
///
/// A set-but-missing `SCENEWORKS_PERSON_DETECTOR_WEIGHTS` is an operator error: it fails
/// loudly (`InvalidPayload`) via [`crate::util::resolve_env_file_pin`] instead of silently
/// falling through to the cache/HF download and loading different weights than the operator
/// asked for (sc-11175/F-011, mirroring the sc-8911 upscaler pin).
pub(crate) fn resolve_detector_weights(settings: &Settings) -> WorkerResult<Option<PathBuf>> {
    if let Some(path) = crate::util::resolve_env_file_pin(
        "SCENEWORKS_PERSON_DETECTOR_WEIGHTS",
        std::env::var_os("SCENEWORKS_PERSON_DETECTOR_WEIGHTS"),
        "the local person-detector weights (MLX safetensors / ONNX export)",
    )? {
        return Ok(Some(path));
    }
    // The installed location: whatever the `person_detector` Model Manager entry cached. That entry
    // declares BOTH platform repos and `retain_downloads_for_os` keeps the one for this OS, so the
    // cfg-selected `DET_REPO`/`DET_REVISION` pair below always names the artifact actually installed.
    if let Some(path) =
        crate::downloads::resolve_hf_component_file(settings, DET_REPO, DET_REVISION, DET_FILE)
    {
        return Ok(Some(path));
    }
    // Legacy pre-migration roots, read-only (AC10 of sc-17598) — nothing writes here any more.
    for sub in ["cache/person-detect", "models/person-detect"] {
        let candidate = settings.data_dir.join(sub).join(DET_FILE);
        if candidate.exists() {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

/// The person-detector weights (MLX safetensors on macOS / ONNX export off-Mac, per
/// `DET_REPO`/`DET_FILE`), or an actionable install error (sc-17629 / sc-17635).
///
/// Replaces the former `ensure_detector_weights`, which downloaded ~80 MB mid-generation. No
/// `DownloadContext` parameter, so this lane cannot fetch (epic 17625).
pub(crate) fn require_detector_weights(settings: &Settings) -> WorkerResult<PathBuf> {
    resolve_detector_weights(settings)?.ok_or_else(|| {
        crate::WorkerError::InvalidPayload(format!(
            "The person detector is not installed. Install \"YOLO11m Person Detector\" from the \
             Model Manager, or point SCENEWORKS_PERSON_DETECTOR_WEIGHTS at a local {DET_FILE}."
        ))
    })
}

#[cfg(test)]
mod tests;
