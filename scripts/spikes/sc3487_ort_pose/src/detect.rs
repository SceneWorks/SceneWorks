//! sc-3487 — Rust DWPose (RTMW whole-body) detector via `ort` (onnxruntime),
//! faithful port of rtmlib's performance preset (yolox_m + rtmw-dw-x-l). Emits the
//! RAW COCO-WholeBody-133 keypoints + SimCC scores per person so the output can be
//! compared 1:1 against scripts/spikes/sc3487_reference.py (the shipped Python
//! detector). Pure CPU pre/post; onnxruntime runs the two models on CPU or CoreML.
//!
//! Pipeline parity notes (matched to rtmlib source):
//!  - YOLOX: letterbox to 640 (ratio=min(640/h,640/w), pad 114, top-left), NO
//!    BGR->RGB swap, NO mean/std; grid-decode strides [8,16,32]; multiclass NMS
//!    score_thr=0.7 nms_thr=0.45 (areas use +1; human mode keeps all classes).
//!  - RTMPose: xyxy->center/scale (padding 1.25) -> aspect-fix to 288/384 ->
//!    top-down affine crop (border 0) -> (px - mean)/std on BGR channels ->
//!    SimCC argmax decode (split 2.0) -> rescale into original px.

use anyhow::{anyhow, Result};
use image::RgbImage;
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DET: usize = 640;
const PW: usize = 288; // pose input width
const PH: usize = 384; // pose input height
const MEAN: [f32; 3] = [123.675, 116.28, 103.53]; // BGR-channel order (rtmlib feeds BGR)
const STD: [f32; 3] = [58.395, 57.12, 57.375];
const SCORE_THR: f32 = 0.7;
const NMS_THR: f32 = 0.45;
const PAD: f32 = 1.25;
const SPLIT: f32 = 2.0;

/// Bilinear sample of a BGR channel from an RgbImage; out-of-bounds -> `border`.
/// `c` is the BGR channel index (0=B,1=G,2=R); RgbImage stores RGB so we map.
#[inline]
fn sample_bgr(img: &RgbImage, x: f32, y: f32, c: usize, border: f32) -> f32 {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let rgb_c = [2usize, 1, 0][c]; // B<-R index2, G<-1, R<-0
    let get = |xi: i64, yi: i64| -> f32 {
        if xi < 0 || yi < 0 || xi >= w || yi >= h {
            border
        } else {
            img.get_pixel(xi as u32, yi as u32)[rgb_c] as f32
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

// ---------------------------------------------------------------------------
// ort glue
// ---------------------------------------------------------------------------

fn build_session(path: &Path, coreml: bool) -> Result<Session> {
    let mut b = Session::builder().map_err(|e| anyhow!(e.to_string()))?;
    if coreml {
        b = b
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
            .map_err(|e| anyhow!(e.to_string()))?;
    }
    b.commit_from_file(path).map_err(|e| anyhow!(e.to_string()))
}

/// Run a single CHW float input; returns each output as (shape, data).
fn run(session: &mut Session, shape: [i64; 4], data: Vec<f32>) -> Result<Vec<(Vec<i64>, Vec<f32>)>> {
    let input = Tensor::from_array((shape.to_vec(), data)).map_err(|e| anyhow!(e.to_string()))?;
    let outputs = session.run(ort::inputs![input]).map_err(|e| anyhow!(e.to_string()))?;
    let n = outputs.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Skip non-f32 outputs (e.g. YOLOX's int64 `labels`); we only consume the
        // float tensors (`dets`, `simcc_x`, `simcc_y`).
        if let Ok((shp, d)) = outputs[i].try_extract_tensor::<f32>() {
            out.push((shp.to_vec(), d.to_vec()));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// YOLOX
// ---------------------------------------------------------------------------

fn yolox_preprocess(img: &RgbImage) -> (Vec<f32>, f32) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let ratio = (DET as f32 / h).min(DET as f32 / w);
    let new_w = (w * ratio) as usize; // int() truncation, matches rtmlib
    let new_h = (h * ratio) as usize;
    let sx = w / new_w as f32; // cv2.resize inverse scale
    let sy = h / new_h as f32;
    // CHW, channel order BGR, pad value 114, no normalization
    let mut chw = vec![114.0f32; 3 * DET * DET];
    for c in 0..3 {
        let plane = c * DET * DET;
        for dy in 0..new_h {
            // cv2 INTER_LINEAR half-pixel mapping
            let src_y = (dy as f32 + 0.5) * sy - 0.5;
            for dx in 0..new_w {
                let src_x = (dx as f32 + 0.5) * sx - 0.5;
                let v = sample_bgr(img, src_x.clamp(0.0, w - 1.0), src_y.clamp(0.0, h - 1.0), c, 0.0);
                chw[plane + dy * DET + dx] = v;
            }
        }
    }
    (chw, ratio)
}

#[derive(Clone, Copy)]
struct Box4 {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
}

fn nms(boxes: &[Box4], thr: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| boxes[b].score.partial_cmp(&boxes[a].score).unwrap());
    let area = |b: &Box4| (b.x2 - b.x1 + 1.0) * (b.y2 - b.y1 + 1.0);
    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];
    for i in 0..order.len() {
        let oi = order[i];
        if suppressed[oi] {
            continue;
        }
        keep.push(oi);
        let bi = &boxes[oi];
        let ai = area(bi);
        for &oj in order.iter().skip(i + 1) {
            if suppressed[oj] {
                continue;
            }
            let bj = &boxes[oj];
            let xx1 = bi.x1.max(bj.x1);
            let yy1 = bi.y1.max(bj.y1);
            let xx2 = bi.x2.min(bj.x2);
            let yy2 = bi.y2.min(bj.y2);
            let iw = (xx2 - xx1 + 1.0).max(0.0);
            let ih = (yy2 - yy1 + 1.0).max(0.0);
            let inter = iw * ih;
            let ovr = inter / (ai + area(bj) - inter);
            if ovr > thr {
                suppressed[oj] = true;
            }
        }
    }
    keep
}

/// Returns person/object bboxes (xyxy, original px) in rtmlib "human" order.
fn yolox_decode(out: &[f32], shape: &[i64], ratio: f32) -> Vec<Box4> {
    // This YOLOX export bakes in NMS: output `dets` is (1, N, 5) = [xyxy, score].
    // rtmlib's `outputs.shape[-1] == 5` branch: boxes /= ratio, keep score > 0.3.
    if *shape.last().unwrap() == 5 {
        let n = shape[1] as usize;
        let mut result = Vec::new();
        for i in 0..n {
            let base = i * 5;
            let score = out[base + 4];
            if score > 0.3 {
                result.push(Box4 {
                    x1: out[base] / ratio,
                    y1: out[base + 1] / ratio,
                    x2: out[base + 2] / ratio,
                    y2: out[base + 3] / ratio,
                    score,
                });
            }
        }
        return result;
    }
    // Fallback: raw YOLOX head (no embedded NMS) — grid decode + multiclass NMS.
    // shape (1, A, 5+C)
    let a = shape[1] as usize;
    let stride_dim = shape[2] as usize;
    let n_cls = stride_dim - 5;
    // build grids/strides in the order meshgrid(arange(w),arange(h)) per level 8,16,32
    let strides = [8usize, 16, 32];
    let mut grid: Vec<(f32, f32, f32)> = Vec::with_capacity(a); // (gx, gy, stride)
    for &s in &strides {
        let hs = DET / s;
        let ws = DET / s;
        for gy in 0..hs {
            for gx in 0..ws {
                grid.push((gx as f32, gy as f32, s as f32));
            }
        }
    }
    assert_eq!(grid.len(), a, "anchor count mismatch");
    let mut per_class: Vec<Vec<Box4>> = vec![Vec::new(); n_cls];
    for i in 0..a {
        let base = i * stride_dim;
        let (gx, gy, st) = grid[i];
        let cx = (out[base] + gx) * st;
        let cy = (out[base + 1] + gy) * st;
        let bw = out[base + 2].exp() * st;
        let bh = out[base + 3].exp() * st;
        let obj = out[base + 4];
        let x1 = (cx - bw / 2.0) / ratio;
        let y1 = (cy - bh / 2.0) / ratio;
        let x2 = (cx + bw / 2.0) / ratio;
        let y2 = (cy + bh / 2.0) / ratio;
        for k in 0..n_cls {
            let score = obj * out[base + 5 + k];
            if score > SCORE_THR {
                per_class[k].push(Box4 { x1, y1, x2, y2, score });
            }
        }
    }
    let mut result = Vec::new();
    for boxes in &per_class {
        if boxes.is_empty() {
            continue;
        }
        for idx in nms(boxes, NMS_THR) {
            result.push(boxes[idx]);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// RTMPose
// ---------------------------------------------------------------------------

fn pose_preprocess(img: &RgbImage, b: &Box4) -> (Vec<f32>, f32, f32, f32, f32) {
    let cx = (b.x1 + b.x2) / 2.0;
    let cy = (b.y1 + b.y2) / 2.0;
    let sw0 = (b.x2 - b.x1) * PAD;
    let sh0 = (b.y2 - b.y1) * PAD;
    let aspect = PW as f32 / PH as f32; // 0.75
    let (sw, sh) = if sw0 > sh0 * aspect {
        (sw0, sw0 / aspect)
    } else {
        (sh0 * aspect, sh0)
    };
    let x0 = cx - sw / 2.0;
    let y0 = cy - sh / 2.0;
    // CHW (1,3,PH,PW), BGR, normalized
    let mut chw = vec![0.0f32; 3 * PH * PW];
    for c in 0..3 {
        let plane = c * PH * PW;
        let m = MEAN[c];
        let sd = STD[c];
        for dy in 0..PH {
            let src_y = y0 + dy as f32 * sh / PH as f32; // affine inverse (no half-pixel for warpAffine)
            for dx in 0..PW {
                let src_x = x0 + dx as f32 * sw / PW as f32;
                let v = sample_bgr(img, src_x, src_y, c, 0.0);
                chw[plane + dy * PW + dx] = (v - m) / sd;
            }
        }
    }
    (chw, cx, cy, sw, sh)
}

/// argmax + max value over a (Wx,) slice.
#[inline]
fn argmax(slice: &[f32]) -> (usize, f32) {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    (bi, bv)
}

/// Decode SimCC outputs -> 133 (x,y,score) in original px.
fn pose_decode(
    simcc_x: &[f32],
    sx_shape: &[i64],
    simcc_y: &[f32],
    cx: f32,
    cy: f32,
    sw: f32,
    sh: f32,
) -> Vec<[f32; 3]> {
    let k = sx_shape[1] as usize; // 133
    let wx = sx_shape[2] as usize; // 576
    let wy = simcc_y.len() / k; // 768
    let x0 = cx - sw / 2.0;
    let y0 = cy - sh / 2.0;
    let mut kpts = Vec::with_capacity(k);
    for j in 0..k {
        let (xloc, mvx) = argmax(&simcc_x[j * wx..(j + 1) * wx]);
        let (yloc, mvy) = argmax(&simcc_y[j * wy..(j + 1) * wy]);
        let val = 0.5 * (mvx + mvy);
        let (mut lx, mut ly) = (xloc as f32, yloc as f32);
        if val <= 0.0 {
            lx = -1.0;
            ly = -1.0;
        }
        let px = (lx / SPLIT) / PW as f32 * sw + x0;
        let py = (ly / SPLIT) / PH as f32 * sh + y0;
        kpts.push([px, py, val]);
    }
    kpts
}

// ---------------------------------------------------------------------------
// orchestration + output
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct PersonOut {
    keypoints: Vec<[f32; 2]>,
    scores: Vec<f32>,
}

#[derive(Serialize)]
struct ImageOut {
    source: String,
    width: u32,
    height: u32,
    device: String,
    det_ms: f64,
    pose_ms: f64,
    bboxes: Vec<[f32; 4]>,
    persons: Vec<PersonOut>,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut images_glob = "/tmp/sc3487/sources/*.png".to_string();
    let mut out_dir = PathBuf::from("/tmp/sc3487/rust");
    let mut coreml = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--images" => {
                images_glob = args[i + 1].clone();
                i += 2;
            }
            "--out" => {
                out_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--device" => {
                coreml = args[i + 1] == "coreml" || args[i + 1] == "mps";
                i += 2;
            }
            _ => i += 1,
        }
    }
    std::fs::create_dir_all(&out_dir)?;
    let device = if coreml { "coreml" } else { "cpu" };

    let home = std::env::var("HOME")?;
    let cache = PathBuf::from(&home).join(".cache/rtmlib/hub/checkpoints");
    let det_path = std::env::var("SCENEWORKS_DWPOSE_DET")
        .map(PathBuf::from)
        .unwrap_or_else(|_| cache.join("yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"));
    let pose_path = std::env::var("SCENEWORKS_DWPOSE_POSE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| cache.join("rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"));

    let t0 = Instant::now();
    let mut det = build_session(&det_path, coreml)?;
    let mut pose = build_session(&pose_path, coreml)?;
    eprintln!("[rust] sessions ready in {:.1}s (device={})", t0.elapsed().as_secs_f64(), device);

    // expand glob manually (no glob dep)
    let dir = Path::new(&images_glob).parent().unwrap_or(Path::new("."));
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "png" || x == "jpg").unwrap_or(false))
        .collect();
    paths.sort();

    let mut index = Vec::new();
    for path in &paths {
        let img = image::open(path)?.to_rgb8();
        let (w, h) = (img.width(), img.height());

        let t1 = Instant::now();
        let (din, ratio) = yolox_preprocess(&img);
        let dout = run(&mut det, [1, 3, DET as i64, DET as i64], din)?;
        let bboxes = yolox_decode(&dout[0].1, &dout[0].0, ratio);
        let det_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let mut persons = Vec::new();
        for b in &bboxes {
            let (pin, cx, cy, sw, sh) = pose_preprocess(&img, b);
            let pout = run(&mut pose, [1, 3, PH as i64, PW as i64], pin)?;
            // disambiguate simcc_x (last dim 576) vs simcc_y (768) by shape
            let (sx_i, sy_i) = if dout_last(&pout[0].0) == (PW * 2) as i64 { (0, 1) } else { (1, 0) };
            let kpts = pose_decode(
                &pout[sx_i].1,
                &pout[sx_i].0,
                &pout[sy_i].1,
                cx,
                cy,
                sw,
                sh,
            );
            persons.push(PersonOut {
                keypoints: kpts.iter().map(|p| [p[0], p[1]]).collect(),
                scores: kpts.iter().map(|p| p[2]).collect(),
            });
        }
        let pose_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let rec = ImageOut {
            source: path.file_name().unwrap().to_string_lossy().into_owned(),
            width: w,
            height: h,
            device: device.to_string(),
            det_ms: (det_ms * 10.0).round() / 10.0,
            pose_ms: (pose_ms * 10.0).round() / 10.0,
            bboxes: bboxes.iter().map(|b| [b.x1, b.y1, b.x2, b.y2]).collect(),
            persons,
        };
        let stem = path.file_stem().unwrap().to_string_lossy();
        std::fs::write(out_dir.join(format!("{stem}.{device}.json")), serde_json::to_string(&rec)?)?;
        eprintln!(
            "[rust] {}: {} person(s) det={:.0}ms pose={:.0}ms bbox0={:?}",
            rec.source,
            rec.bboxes.len(),
            det_ms,
            pose_ms,
            rec.bboxes.first().map(|b| [b[0].round(), b[1].round(), b[2].round(), b[3].round()])
        );
        index.push(rec.source.clone());
    }
    eprintln!("[rust] DONE {} images -> {}", index.len(), out_dir.display());
    Ok(())
}

#[inline]
fn dout_last(shape: &[i64]) -> i64 {
    *shape.last().unwrap()
}
