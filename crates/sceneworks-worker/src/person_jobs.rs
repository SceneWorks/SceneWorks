//! YOLO11 person detection on the Rust worker (epic 3482, sc-3488 slice 1 / sc-3633).
//!
//! Ports the Python `scene_worker/person_adapters.py` `_UltralyticsDetector`
//! (Ultralytics `yolo11m.pt`, COCO class 0) to Rust so the Replace-Person
//! detection step runs on a Python-free Mac. Inference uses the `ort`
//! (onnxruntime) scaffold proven by `pose_jobs.rs` (sc-3487): a process-wide
//! cached `Session` and all `ort` objects confined to a `spawn_blocking`
//! closure. Unlike `pose_jobs`, this runs on the **CPU EP by default** — the
//! CoreML EP hangs in `commit_from_file` on the Ultralytics YOLO11 export (see
//! `Detector::load`); CoreML is opt-in via `SCENEWORKS_PERSON_DETECTOR_COREML=1`.
//!
//! macOS-only in practice: it gates with `pose_jobs`, and the Python Ultralytics
//! path stays the Windows/Linux detector. The pure detector
//! math (letterbox / decode / NMS / box normalization) is unit-tested without
//! the onnx weights; only the onnxruntime inference itself is gated.
//!
//! Pipeline (matched to the Ultralytics YOLO11 ONNX export — verified against the
//! real `yolo11m.onnx` + `ultralytics.predict`):
//!  - input `images` (1,3,640,640) f32: letterbox to 640 (ratio=min(640/w,640/h),
//!    pad 114 centered), RGB channel order, divided by 255. cv2 INTER_LINEAR
//!    half-pixel sampling.
//!  - output `output0` (1,84,8400), channel-major: rows 0..4 = cx,cy,w,h in
//!    letterbox px; rows 4..84 = 80 sigmoid class scores in [0,1] (no separate
//!    objectness). Person = class 0 → channel 4.
//!  - decode: keep anchors whose person score > conf, cx/cy/w/h → xyxy, subtract
//!    the letterbox pad and divide by the ratio back into original px, clamp to
//!    the frame, then greedy NMS at the Ultralytics default IoU 0.7.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use image::RgbImage;
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use serde_json::{json, Value};

use crate::{Settings, WorkerError, WorkerResult};

/// Square detector input edge (the YOLO11 export is fixed 640×640).
const DET: usize = 640;
/// COCO "person" class index, matching Python `PERSON_CLASS_INDEX`.
const PERSON_CLASS: usize = 0;
/// Letterbox padding value (Ultralytics default), pre-`/255`.
const PAD_VALUE: f32 = 114.0;
/// Greedy-NMS IoU threshold — Ultralytics `predict` default (`iou=0.7`).
const NMS_IOU: f32 = 0.7;
/// The detector onnx filename in the app cache / model dir.
const DET_FILE: &str = "yolo11m.onnx";

// ---------------------------------------------------------------------------
// pure detector math (unit-tested without weights)
// ---------------------------------------------------------------------------

/// One person box in the *original frame's* pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl Detection {
    fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }
    fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }
    fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

/// Letterbox geometry: the resize ratio and the centered left/top pad (px) in
/// the 640 input. `un_letterbox` inverts it back to original-frame px.
#[derive(Clone, Copy, Debug)]
struct Letterbox {
    ratio: f32,
    pad_x: f32,
    pad_y: f32,
}

impl Letterbox {
    fn compute(w: u32, h: u32) -> Self {
        let (w, h) = (w as f32, h as f32);
        let ratio = (DET as f32 / w).min(DET as f32 / h);
        let new_w = (w * ratio).round();
        let new_h = (h * ratio).round();
        // Ultralytics splits the pad in half with a -0.1 bias on the lead edge.
        let pad_x = ((DET as f32 - new_w) / 2.0 - 0.1).round();
        let pad_y = ((DET as f32 - new_h) / 2.0 - 0.1).round();
        Self {
            ratio,
            pad_x,
            pad_y,
        }
    }

    /// Map a letterbox-space coordinate back into original-frame px.
    fn un_x(&self, x: f32) -> f32 {
        (x - self.pad_x) / self.ratio
    }
    fn un_y(&self, y: f32) -> f32 {
        (y - self.pad_y) / self.ratio
    }
}

/// Bilinear sample of an RGB channel (0=R,1=G,2=B) at (x,y); out-of-bounds →
/// `border`. cv2 INTER_LINEAR half-pixel convention is applied by the caller.
#[inline]
fn sample_rgb(img: &RgbImage, x: f32, y: f32, c: usize, border: f32) -> f32 {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let get = |xi: i64, yi: i64| -> f32 {
        if xi < 0 || yi < 0 || xi >= w || yi >= h {
            border
        } else {
            img.get_pixel(xi as u32, yi as u32)[c] as f32
        }
    };
    let v00 = get(x0, y0);
    let v10 = get(x0 + 1, y0);
    let v01 = get(x0, y0 + 1);
    let v11 = get(x0 + 1, y0 + 1);
    let top = v00 * (1.0 - fx) + v10 * fx;
    let bot = v01 * (1.0 - fx) + v11 * fx;
    top * (1.0 - fy) + bot * fy
}

/// Build the (1,3,640,640) RGB `/255` letterboxed input and return the geometry.
fn preprocess(img: &RgbImage) -> (Vec<f32>, Letterbox) {
    let lb = Letterbox::compute(img.width(), img.height());
    let (w, h) = (img.width() as f32, img.height() as f32);
    let new_w = (w * lb.ratio).round();
    let new_h = (h * lb.ratio).round();
    let sx = w / new_w.max(1.0);
    let sy = h / new_h.max(1.0);
    let new_w = new_w as usize;
    let new_h = new_h as usize;
    let pad_x = lb.pad_x as usize;
    let pad_y = lb.pad_y as usize;

    let mut chw = vec![PAD_VALUE / 255.0; 3 * DET * DET];
    for c in 0..3 {
        let plane = c * DET * DET;
        for dy in 0..new_h {
            let src_y = (dy as f32 + 0.5) * sy - 0.5; // cv2 INTER_LINEAR half-pixel
            let row = plane + (dy + pad_y) * DET + pad_x;
            for dx in 0..new_w {
                let src_x = (dx as f32 + 0.5) * sx - 0.5;
                let v = sample_rgb(
                    img,
                    src_x.clamp(0.0, w - 1.0),
                    src_y.clamp(0.0, h - 1.0),
                    c,
                    0.0,
                );
                chw[row + dx] = v / 255.0;
            }
        }
    }
    (chw, lb)
}

/// Decode the (1,84,8400) channel-major output into person boxes (original px),
/// pre-NMS. `data` is laid out as `data[channel * anchors + anchor]`.
fn decode(
    data: &[f32],
    shape: &[i64],
    lb: &Letterbox,
    conf: f32,
    frame_w: u32,
    frame_h: u32,
) -> Vec<Detection> {
    let channels = shape[1] as usize; // 84 = 4 box + 80 classes
    let anchors = shape[2] as usize; // 8400
    if channels < 5 {
        return Vec::new();
    }
    let score_ch = 4 + PERSON_CLASS;
    let (fw, fh) = (frame_w as f32, frame_h as f32);
    let mut out = Vec::new();
    for a in 0..anchors {
        let score = data[score_ch * anchors + a];
        if score <= conf {
            continue;
        }
        let cx = data[a];
        let cy = data[anchors + a];
        let bw = data[2 * anchors + a];
        let bh = data[3 * anchors + a];
        let x1 = lb.un_x(cx - bw / 2.0).clamp(0.0, fw);
        let y1 = lb.un_y(cy - bh / 2.0).clamp(0.0, fh);
        let x2 = lb.un_x(cx + bw / 2.0).clamp(0.0, fw);
        let y2 = lb.un_y(cy + bh / 2.0).clamp(0.0, fh);
        out.push(Detection {
            x1,
            y1,
            x2,
            y2,
            score,
        });
    }
    out
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let union = a.area() + b.area() - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Greedy non-max suppression, score-descending, single (person) class.
fn nms(mut dets: Vec<Detection>, iou_thr: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep: Vec<Detection> = Vec::new();
    for det in dets {
        if keep.iter().all(|k| iou(k, &det) <= iou_thr) {
            keep.push(det);
        }
    }
    keep
}

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

// ---------------------------------------------------------------------------
// onnxruntime detector (cached process-wide like Python's lazy model load)
// ---------------------------------------------------------------------------

struct Detector {
    session: Session,
    device: &'static str,
}

static DETECTOR: OnceLock<Mutex<Option<Detector>>> = OnceLock::new();

fn ort_err<R>(e: ort::Error<R>) -> WorkerError {
    WorkerError::InvalidPayload(format!("onnxruntime: {e}"))
}

fn build_session(path: &Path, coreml: bool) -> WorkerResult<Session> {
    let mut b = Session::builder().map_err(ort_err)?;
    if coreml {
        b = b
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
            .map_err(ort_err)?;
    }
    b.commit_from_file(path).map_err(ort_err)
}

impl Detector {
    /// Build the cached detector. Unlike `pose_jobs` (YOLOX), the CoreML EP
    /// *hangs* indefinitely in `commit_from_file` on the Ultralytics YOLO11
    /// export — it never returns an error, so a try-CoreML-then-fall-back is
    /// unsafe (it would deadlock the job). The CPU EP runs YOLO11m fine and is
    /// what the Python reference (onnxruntime `CPUExecutionProvider`) used; for
    /// a single representative frame / 2-fps tracking it's well within budget.
    /// CoreML stays opt-in via `SCENEWORKS_PERSON_DETECTOR_COREML=1` for future
    /// investigation (export-opset / MLProgram tweaks); see sc-3633 notes.
    fn load(path: &Path) -> WorkerResult<Self> {
        let try_coreml = std::env::var("SCENEWORKS_PERSON_DETECTOR_COREML")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if try_coreml {
            if let Ok(session) = build_session(path, true) {
                return Ok(Self {
                    session,
                    device: "coreml",
                });
            }
        }
        Ok(Self {
            session: build_session(path, false)?,
            device: "cpu",
        })
    }

    fn detect(&mut self, img: &RgbImage, conf: f32) -> WorkerResult<Vec<Detection>> {
        let (input, lb) = preprocess(img);
        let tensor = Tensor::from_array(([1_i64, 3, DET as i64, DET as i64].to_vec(), input))
            .map_err(ort_err)?;
        let outputs = self.session.run(ort::inputs![tensor]).map_err(ort_err)?;
        let (shp, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        let shape = shp.to_vec();
        let raw = decode(data, &shape, &lb, conf, img.width(), img.height());
        Ok(nms(raw, NMS_IOU))
    }
}

/// Person detections for one frame, plus the device the model ran on.
pub(crate) struct DetectResult {
    pub width: u32,
    pub height: u32,
    pub detections: Vec<Detection>,
    pub device: &'static str,
}

/// Blocking person detection on a single rendered frame. All `ort` state lives
/// inside this call (built once, cached process-wide) and never crosses an
/// await; invoke via `spawn_blocking`.
pub(crate) fn detect_people_blocking(
    onnx_path: PathBuf,
    image_path: PathBuf,
    conf: f32,
) -> WorkerResult<DetectResult> {
    let img = image::open(&image_path)
        .map_err(|e| WorkerError::InvalidPayload(format!("person frame open: {e}")))?
        .to_rgb8();
    let (width, height) = (img.width(), img.height());

    let cell = DETECTOR.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().expect("person detector mutex poisoned");
    if guard.is_none() {
        *guard = Some(Detector::load(&onnx_path)?);
    }
    let detector = guard.as_mut().expect("detector loaded");
    let device = detector.device;
    let detections = detector.detect(&img, conf)?;
    Ok(DetectResult {
        width,
        height,
        detections,
        device,
    })
}

// ---------------------------------------------------------------------------
// weights resolution (download-on-first-use lands in slice 4 / sc-3636)
// ---------------------------------------------------------------------------

/// Resolve the detector onnx: explicit env pin (`SCENEWORKS_PERSON_DETECTOR_ONNX`),
/// then the app cache `<data_dir>/cache/person-detect/`, then the model dir
/// `<data_dir>/models/person-detect/`. Returns `None` when no weight is present
/// (slice 4 adds download-on-first-use so the Mac is provisioned automatically).
pub(crate) fn resolve_detector_onnx(settings: &Settings) -> Option<PathBuf> {
    if let Ok(pinned) = std::env::var("SCENEWORKS_PERSON_DETECTOR_ONNX") {
        let path = PathBuf::from(pinned);
        if path.exists() {
            return Some(path);
        }
    }
    for sub in ["cache/person-detect", "models/person-detect"] {
        let candidate = settings.data_dir.join(sub).join(DET_FILE);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

#[cfg(test)]
mod tests;
