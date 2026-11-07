//! DWPose orchestration, model resolution, pose records and skeleton rendering.
//!
//! The pinned runtime's CV engine owns YOLOX/RTMW execution, numeric preprocessing,
//! SimCC decoding and OpenPose conversion. The worker retains its paired-session
//! cache, input resolution, between-image cancellation, render/metadata policy,
//! and pose-library persistence. ORT uses CoreML on macOS and CUDA off-Mac, with
//! the original paired CPU fallback and instrumented `model_io` loader.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::asset_media::resolve_asset_media;
use crate::cv::pose::{squareify, wholebody_to_openpose, Detector, PoseRecord};
use image::RgbImage;
use serde_json::{json, Value};

use crate::openpose_skeleton::{body_stickwidth, draw_wholebody, Keypoint};
use crate::util::{resolve_env_file_pin, resolved_cache_paths_match};
use crate::{
    heartbeat, normalize_app_managed_cache_path, optional_payload_string, progress_payload,
    run_blocking_with_heartbeat, update_job, ApiClient, Settings, WorkerError, WorkerResult,
};
use sceneworks_core::contracts::{JobSnapshot, JobStatus, JsonObject, ProgressStage, WorkerStatus};
use sceneworks_core::project_store::{ProjectStore, POSE_UPLOADS_CACHE_DIR};

// ---------------------------------------------------------------------------
// detector constants (rtmlib performance preset)
// ---------------------------------------------------------------------------

const DET_FILE: &str = "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx";
const POSE_FILE: &str = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx";
/// SceneWorks-owned re-host of the two DWPose ONNX graphs (sc-17634, epic 17625). Public;
/// declared by the `dwpose_pose_detector` catalog entry and installed from the Model Manager —
/// NOT downloaded on first use any more.
///
/// # Why a re-host exists at all
///
/// Until sc-17634 this lane fetched two `.zip` bundles straight from `download.openmmlab.com`
/// and unpacked them into `<data_dir>/cache/dwpose/` — the only non-Hugging-Face weight download
/// in the product. That could never be installed from the Model Manager: a download row is
/// hard-wired to `provider: "huggingface"` in four independent places (`is_supported_model_download`,
/// the manifest schema's `required: ["provider","repo"]`, `install_state_for`, `model_artifact_paths`),
/// so a direct-URL row is silently invisible rather than an error. Re-hosting the extracted graphs
/// removes the exception instead of building a second install path for one model.
///
/// The files here are the `end2end.onnx` entry of each upstream archive, byte-for-byte — no
/// re-export, no conversion, no quantization. The repo's model card records both archive URLs,
/// their SHA-256s, and the extraction map, so the mirror is reproducible from upstream.
const DWPOSE_REPO: &str = "SceneWorks/dwpose-onnx";
/// Pinned DWPose ONNX revision. Even though `SceneWorks/dwpose-onnx` is a first-party repo,
/// resolving the mutable `main` branch would mean a re-push (or a compromised token) could
/// silently swap the graphs we load, so pin the exact commit — the same defense-in-depth as
/// [`crate::upscale_jobs`]'s Real-ESRGAN and SeedVR2 pins.
///
/// This replaces the sc-8879 zip digests, and is a strictly stronger integrity story rather than
/// a weaker one: the digests only ever covered a transfer this code no longer performs, whereas
/// the Model Manager verifies every file against Hugging Face's `lfs.oid` at install time, and
/// this pin then binds the load to that exact snapshot. (Both LFS oids happen to equal the
/// SHA-256s of the graphs the old `extract_onnx` produced, so an existing install and a fresh one
/// hold identical bytes.)
///
/// The `dwpose_pose_detector` catalog entry declares this SAME revision, so the Model Manager
/// installs exactly the snapshot [`require_dwpose_weights`] then reads;
/// `manifest_declares_the_same_dwpose_pin_the_loader_reads` fails if the two drift.
pub(crate) const DWPOSE_REVISION: &str = "c330a9609201343576b8b69d222e38c1486d2854";
/// Operator overrides naming a local `.onnx`, one per graph. Both are **file** pins and both keep
/// sc-8911's hard failure: set-but-missing is an operator error (a typo silently loading different
/// weights is the failure mode that rule exists to prevent), not a fall-through.
///
/// Unlike [`crate::upscale_jobs`]'s `SEEDVR2_DIR_PINS`, no strictness question arises here. That
/// pair needed splitting because `SCENEWORKS_SEEDVR2_DIR` had been a *staging destination* the
/// pre-migration code created and downloaded into, so an empty one was legal under its old
/// meaning. Neither DWPose key was ever a staging path — both have always named an existing file
/// through [`resolve_env_file_pin`] — so migrating changes nothing about how either behaves.
const DWPOSE_FILE_PINS: [(&str, &str); 2] = [
    ("SCENEWORKS_DWPOSE_DET", DET_FILE),
    ("SCENEWORKS_DWPOSE_POSE", POSE_FILE),
];
/// Pre-migration weight roots, read-only (AC10 of sc-17598), in the order the deleted `ensure_one`
/// consulted them before sc-17634 so an existing install resolves to exactly the file it did before.
///
/// * `<data_dir>/cache/dwpose/` — where the deleted download+unzip path wrote (~300 MB for the
///   pair). App-managed, so it is joined onto `settings.data_dir`.
/// * `~/.cache/rtmlib/hub/checkpoints/` — rtmlib's own cache, present on dev machines and on
///   installs that predate the Rust port. NOT app-managed: it is keyed off `$HOME`, which is why
///   it cannot be expressed as a `data_dir` subpath like the other legacy roots in this epic.
///
/// Nothing writes to either any more, so they only drain; reclaiming them is sc-17636 (S11).
const DWPOSE_LEGACY_DATA_DIR_ROOT: &str = "cache/dwpose";
const DWPOSE_LEGACY_RTMLIB_ROOT: &str = ".cache/rtmlib/hub/checkpoints";
const DETECTOR_ID: &str = "rtmw-dw-x-l/ort";

/// The ONNX Runtime minor version SceneWorks actually ships. Four pin sites, all `1.26.0`:
///
/// * `apps/desktop/src/cuda_provision.rs` — the Windows `onnxruntime-gpu` wheel + `REDIST_VERSION`;
/// * `apps/desktop/src/linux_cuda_provision.rs` — the Linux sibling;
/// * `apps/desktop/scripts/stage-onnxruntime.py` — the macOS CoreML dylib bundled into the app;
/// * `docker/rust.Dockerfile` — `ONNXRUNTIME_GPU_VERSION`.
///
/// Bump this only together with all four.
const PROVISIONED_ONNXRUNTIME_MINOR: u32 = 26;

/// `ort` must not request a newer ONNX Runtime API than the runtime we provision.
///
/// `ort`'s `load-dynamic` loader compares [`ort::MINOR_VERSION`] (= `ort_sys::ORT_API_VERSION`,
/// selected by whichever `api-N` feature is enabled) against the dylib's own minor and returns
/// `LoadError::BadVersion` when the dylib is LOWER. Nothing about that mismatch is otherwise
/// visible before shipping: it surfaces only when a user runs a pose / person / upscale job, as
///
/// > Failed to load ONNX Runtime dylib: BadVersion { version_str: "1.26.0", … }
///
/// which is exactly what shipped when dependabot's group bump (#2092) moved `ort` rc.12 -> rc.13
/// and, with it, the default `api-24` -> `api-27` — every `ort` surface on every platform broken by
/// a bump that compiled clean and passed CI.
///
/// So it is a `const` assertion, not a `#[test]`: this is checked by every `cargo build`/`check` of
/// this module, including the candle typecheck lane, which never RUNS tests (it cannot link without
/// a real libcuda). A test would have been silent in precisely the lane that first compiles this
/// code off-Mac. `<=` and not `==`: shipping a runtime NEWER than the requested API is fine (ORT's
/// C API is forward compatible and `ort` only logs); requesting an API newer than the runtime is
/// the fatal direction.
///
/// # Fixing a failure here
///
/// The tempting fix — bump the provisioned wheel — is usually the wrong one. onnxruntime-gpu 1.27
/// moved its CUDA execution provider to CUDA 13 (`nvidia-*-cu13`), while every other DLL the
/// provisioners stage, and cudarc/candle besides, is CUDA 12.9. Raising
/// [`PROVISIONED_ONNXRUNTIME_MINOR`] means migrating the whole GPU runtime, not swapping one wheel.
/// Pinning `ort`'s `api-N` feature back down (workspace `Cargo.toml`) is the cheap direction.
const _: () = assert!(
    ort::MINOR_VERSION <= PROVISIONED_ONNXRUNTIME_MINOR,
    "`ort` requests a newer ONNX Runtime API than SceneWorks provisions (see \
     PROVISIONED_ONNXRUNTIME_MINOR above): every pose / person / upscale job would fail at runtime \
     with `BadVersion`. Pin the `ort` api-N feature down in the workspace Cargo.toml, or bump all \
     four provisioning sites — but read the CUDA-13 note above before choosing the latter."
);

/// Default per-keypoint confidence floor for rendering/thresholding. rtmlib's RTMW
/// SimCC scores are NOT in `[0,1]` (good keypoints observed ~4–8 in the spike), so
/// this is a render/threshold knob, not a probability; raw scores are preserved in
/// the result. Mirrors Python `DEFAULT_POSE_MIN_CONF`. Owned by [`crate::control_preprocess`] so
/// the pose preprocessor registry and this job share one floor.
pub(crate) const DEFAULT_POSE_MIN_CONF: f64 = crate::control_preprocess::DEFAULT_POSE_MIN_CONF;

/// Terminal cancel message for the pose-detect job. Shared between the
/// `run_blocking_with_heartbeat` keepalive (which posts it when the blocking task finishes AFTER
/// the flag tripped) and the in-task between-iteration checks below (whose typed `Canceled`
/// carries the same message), so the terminal status is identical whichever side observes the trip
/// first (sc-9123).
const POSE_CANCEL_MESSAGE: &str = "Pose detection canceled by user.";

/// Between-iteration cooperative cancel check for the two worker-side pose loops (sc-9123): the
/// DWPose batch-detect loop and the skeleton render loop both live in worker code (not behind an
/// engine surface), so a user cancel tripped by the keepalive's poll bails between images/persons
/// with the TYPED `Canceled` — which `run_blocking_with_heartbeat` maps to the terminal `Canceled`
/// post instead of failing the job.
fn bail_if_canceled(cancel: &gen_core::CancelFlag) -> WorkerResult<()> {
    if cancel.is_cancelled() {
        return Err(WorkerError::Canceled(POSE_CANCEL_MESSAGE.to_owned()));
    }
    Ok(())
}

/// Render-ready keypoints: drop points below the confidence floor (→ `None`).
fn thresholded(group: &[[f32; 3]], min_conf: f64) -> Vec<Keypoint> {
    group
        .iter()
        .map(|p| {
            if (p[2] as f64) < min_conf {
                None
            } else {
                Some((p[0], p[1]))
            }
        })
        .collect()
}

/// Bbox over the thresholded points: `[minx, miny, maxx, maxy]` or `None`.
fn bbox(groups: &[&[[f32; 3]]], min_conf: f64) -> Option<[f32; 4]> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for g in groups {
        for p in *g {
            if (p[2] as f64) >= min_conf {
                xs.push(p[0]);
                ys.push(p[1]);
            }
        }
    }
    if xs.is_empty() {
        return None;
    }
    let (mut x0, mut x1, mut y0, mut y1) = (xs[0], xs[0], ys[0], ys[0]);
    for &x in &xs {
        x0 = x0.min(x);
        x1 = x1.max(x);
    }
    for &y in &ys {
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    Some([x0, y0, x1, y1])
}

fn mean_conf(group: &[[f32; 3]]) -> f64 {
    if group.is_empty() {
        return 0.0;
    }
    let sum: f64 = group.iter().map(|p| p[2] as f64).sum();
    (sum / group.len() as f64 * 1000.0).round() / 1000.0
}

/// front / back / profile from the OpenPose-18 head keypoints. Mirrors `_facing`.
fn facing(body: &[[f32; 3]], min_conf: f64) -> &'static str {
    let ok = |i: usize| (body[i][2] as f64) >= min_conf;
    let (nose, r_eye, l_eye, r_ear, l_ear) = (ok(0), ok(14), ok(15), ok(16), ok(17));
    if !nose && !r_eye && !l_eye {
        return "back";
    }
    if r_ear && l_ear {
        return "front";
    }
    if r_ear != l_ear {
        return "profile";
    }
    "front"
}

fn record_to_json(group: &[[f32; 3]]) -> Value {
    Value::Array(
        group
            .iter()
            .map(|p| json!([p[0], p[1], p[2]]))
            .collect::<Vec<_>>(),
    )
}

// ---------------------------------------------------------------------------
// onnxruntime detector (cached process-wide like Python `_DETECTOR_CACHE`)
// ---------------------------------------------------------------------------

static DETECTOR: OnceLock<Mutex<Option<(PathBuf, PathBuf, Detector)>>> = OnceLock::new();

/// One source image's raw detection (post-inference, pre-SceneWorks-conversion).
struct RawSource {
    width: u32,
    height: u32,
    people: Vec<Vec<[f32; 3]>>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CatalogPosePerson {
    /// SceneWorks OpenPose-18 body keypoints normalized to the source frame.
    pub body_keypoints: Vec<[f32; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CatalogPoseResult {
    pub people: Vec<CatalogPosePerson>,
}

/// Detect every person's OpenPose-18 body keypoints for one catalog image,
/// reusing the same process-wide DWPose sessions as pose jobs and control
/// preprocessing. Blocking; callers must keep it off async runtime threads.
pub(crate) fn detect_catalog_poses_blocking(
    det_path: &Path,
    pose_path: &Path,
    image_path: &Path,
) -> WorkerResult<CatalogPoseResult> {
    let image = crate::image_decode::decode_image_any(image_path)
        .map_err(|error| WorkerError::InvalidPayload(format!("pose source open: {error}")))?
        .to_rgb8();
    let (width, height) = (image.width(), image.height());
    let cell = DETECTOR.get_or_init(|| Mutex::new(None));
    let mut guard = crate::cache_thread::lock_resetting(cell, |slot| *slot = None);
    let detector = &mut crate::cache_thread::get_or_load(
        &mut guard,
        |(cached_det, cached_pose, _)| {
            resolved_cache_paths_match(
                [cached_det.as_path(), cached_pose.as_path()],
                [det_path, pose_path],
            )
        },
        || {
            Ok((
                det_path.to_path_buf(),
                pose_path.to_path_buf(),
                Detector::load(det_path, pose_path, &mut crate::cv::ModelIoLoader)?,
            ))
        },
    )?
    .2;
    let people = detector
        .detect(&image)?
        .iter()
        .map(|keypoints| {
            wholebody_to_openpose(keypoints, width as f32, height as f32).map(|record| {
                CatalogPosePerson {
                    body_keypoints: record.keypoints,
                }
            })
        })
        .collect::<crate::cv::Result<Vec<_>>>()?;
    Ok(CatalogPoseResult { people })
}

/// Blocking detection over a batch of resolved image paths. All `ort` objects live
/// inside this closure (built once, then cached process-wide) and never cross an
/// await. `None` entries are unreadable images.
fn detect_batch(
    det_path: PathBuf,
    pose_path: PathBuf,
    images: Vec<Option<PathBuf>>,
    cancel: &gen_core::CancelFlag,
) -> WorkerResult<(Vec<Option<RawSource>>, &'static str)> {
    // A cancel that lands before the (long) cold detector load starts skips the load entirely.
    bail_if_canceled(cancel)?;
    let cell = DETECTOR.get_or_init(|| Mutex::new(None));
    let mut guard = crate::cache_thread::lock_resetting(cell, |slot| *slot = None);
    let detector = &mut crate::cache_thread::get_or_load(
        &mut guard,
        |(cached_det, cached_pose, _)| {
            resolved_cache_paths_match(
                [cached_det.as_path(), cached_pose.as_path()],
                [det_path.as_path(), pose_path.as_path()],
            )
        },
        || {
            Ok((
                det_path.clone(),
                pose_path.clone(),
                Detector::load(&det_path, &pose_path, &mut crate::cv::ModelIoLoader)?,
            ))
        },
    )?
    .2;
    let device = detector.device;
    let mut out = Vec::with_capacity(images.len());
    for path in images {
        // sc-9123: this multi-image batch loop is worker code, so a user cancel (tripped by the
        // keepalive's poll) bails between images with the TYPED `Canceled` — each per-image forward
        // stays a bounded uninterruptible unit, but the batch as a whole is cancellable.
        bail_if_canceled(cancel)?;
        let Some(path) = path else {
            out.push(None);
            continue;
        };
        let img = match crate::image_decode::decode_image_any(&path) {
            Ok(img) => img.to_rgb8(),
            Err(_) => {
                out.push(None);
                continue;
            }
        };
        let (width, height) = (img.width(), img.height());
        let people = detector.detect(&img)?;
        out.push(Some(RawSource {
            width,
            height,
            people,
        }));
    }
    Ok((out, device))
}

/// Detect the dominant person in `image` and render an OpenPose skeleton control image,
/// reusing the SAME cached DWPose detector + `wholebody_to_openpose` conversion + squareify +
/// `draw_wholebody` renderer the `pose_detect` job uses. This is the pose entry the
/// ControlNet-Training-Studio preprocessor registry calls (sc-10160): sharing the render path
/// makes the train-prep skeleton convention identical to the inference lanes' by construction
/// (the pose lesson — train/infer skeletons must be the same render convention).
///
/// `components` selects which keypoint groups are drawn (8459 Config-1 = body-18 + head only:
/// [`crate::control_preprocess::PoseComponents::Body`]). The skeleton is drawn on a `max(w,h)`
/// SQUARE canvas (aspect-canonical, matching `squareify`'s stored-pose convention). The
/// largest-person-area figure wins when several are detected; a source with no detected person
/// yields an all-black canvas (a valid empty condition, not an error).
///
/// Blocking (onnx forward + raster); callers on an async runtime should wrap it in
/// `spawn_blocking`, exactly as the batch job does for its detect + render loops.
///
pub(crate) fn detect_and_render_skeleton(
    det_path: &Path,
    pose_path: &Path,
    image: &RgbImage,
    components: crate::control_preprocess::PoseComponents,
    min_conf: f64,
) -> WorkerResult<RgbImage> {
    let (w, h) = (image.width() as f32, image.height() as f32);
    let side = image.width().max(image.height());
    let stick = body_stickwidth(side, side);

    // Process-wide cached detector — the same `DETECTOR` static + EP-with-CPU-fallback load the
    // batch path uses, so the studio and the pose_detect job share one loaded model.
    let cell = DETECTOR.get_or_init(|| Mutex::new(None));
    let mut guard = crate::cache_thread::lock_resetting(cell, |slot| *slot = None);
    let detector = &mut crate::cache_thread::get_or_load(
        &mut guard,
        |(cached_det, cached_pose, _)| {
            resolved_cache_paths_match(
                [cached_det.as_path(), cached_pose.as_path()],
                [det_path, pose_path],
            )
        },
        || {
            Ok((
                det_path.to_path_buf(),
                pose_path.to_path_buf(),
                Detector::load(det_path, pose_path, &mut crate::cv::ModelIoLoader)?,
            ))
        },
    )?
    .2;
    let people = detector.detect(image)?;

    // Convert + squareify each person, keep the largest-area one (same ordering rule as the
    // batch render's `sort_by area desc`, reduced to an arg-max since we render a single figure).
    let mut best: Option<(f32, PoseRecord)> = None;
    for kps in &people {
        let rec = squareify(&wholebody_to_openpose(kps, w, h)?, w, h);
        let area = bbox(
            &[&rec.keypoints, &rec.hands[0], &rec.hands[1], &rec.face],
            min_conf,
        )
        .map_or(0.0, |b| (b[2] - b[0]) * (b[3] - b[1]));
        if best.as_ref().is_none_or(|(a, _)| area > *a) {
            best = Some((area, rec));
        }
    }

    let Some((_, rec)) = best else {
        return Ok(RgbImage::new(side, side));
    };

    let body = thresholded(&rec.keypoints, min_conf);
    let hands = components.wants_hands().then(|| {
        [
            thresholded(&rec.hands[0], min_conf),
            thresholded(&rec.hands[1], min_conf),
        ]
    });
    let face = components
        .wants_face()
        .then(|| thresholded(&rec.face, min_conf));
    Ok(draw_wholebody(
        side,
        side,
        &body,
        hands.as_ref().map(|h| &h[..]),
        face.as_deref(),
        stick,
    ))
}

// ---------------------------------------------------------------------------
// weights resolution (cache-only; installed from the Model Manager)
// ---------------------------------------------------------------------------

/// The two DWPose onnx weight paths `(detector, pose)`, or an actionable install error
/// (sc-17634 / S10, sc-17635) — the public seam the ControlNet Training Studio job (epic 10159,
/// sc-10162) uses to build a [`crate::control_preprocess::PreprocessResources`] before rendering
/// pose conditions, and the same resolver `run_pose_detect_job` uses, so the studio and the
/// pose-detect job provision identical weights.
///
/// Taking `&Settings` and **no `DownloadContext`** is the enforcement, not an aesthetic choice:
/// this lane cannot reach the network any more, by construction rather than by convention
/// (epic 17625). It replaces the former `ensure_dwpose_weights`, which fetched ~300 MB of zip
/// bundles mid-job and unpacked them into a destination the Models screen can neither size nor
/// delete.
pub(crate) fn require_dwpose_weights(settings: &Settings) -> WorkerResult<(PathBuf, PathBuf)> {
    // Destructured, not iterated: the array's LENGTH is part of its type, so emptying or extending
    // `DWPOSE_FILE_PINS` is a compile error here rather than a silently-shorter loop. That is the
    // sc-17632 lesson — a test that walked a legacy-dirs constant bare kept passing when the
    // constant was emptied — answered at the type level instead of only in a test.
    let [(det_key, det_file), (pose_key, pose_file)] = DWPOSE_FILE_PINS;
    let det = require_dwpose_graph(det_key, det_file, settings)?;
    let pose = require_dwpose_graph(pose_key, pose_file, settings)?;
    Ok((det, pose))
}

/// Resolve one already-present DWPose graph, **cache-only** (sc-17634, epic 17625). Order is
/// unchanged from the pre-migration `ensure_one`, minus its download tail:
///
/// 1. the explicit env pin for this graph ([`DWPOSE_FILE_PINS`]);
/// 2. the **HF cache**, where the `dwpose_pose_detector` Model Manager install puts it — the
///    pinned `snapshots/<sha>/` of [`DWPOSE_REPO`] at [`DWPOSE_REVISION`];
/// 3. the legacy `<data_dir>/cache/dwpose/` copy, read-only;
/// 4. the legacy `~/.cache/rtmlib/hub/checkpoints/` copy, read-only.
///
/// Returns `Ok(None)` when nothing is staged; [`require_dwpose_graph`] turns that into the install error.
///
/// # Each graph resolves independently, on purpose
///
/// Unlike SeedVR2's DiT+VAE pair ([`crate::upscale_jobs::resolve_seedvr2_checkpoint_dir`]), the
/// detector and the pose model are **not** a torn-pair hazard: they are two independent
/// single-file `ort` sessions, not one directory handed to an engine, and each filename names
/// exactly one artifact whose bytes are identical in every root. So a box with the detector in
/// `cache/dwpose/` and the pose model in the HF cache resolves both and loads correct weights —
/// which is precisely the pre-migration behaviour, preserved.
fn resolve_dwpose_graph(
    env_key: &str,
    file: &str,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    // A set-but-missing env pin is an operator error: fail loudly instead of silently falling
    // through to the cache and loading different weights than the operator asked for (sc-8911).
    if let Some(path) = resolve_env_file_pin(env_key, std::env::var_os(env_key), file)? {
        return Ok(Some(path));
    }
    // The installed location: whatever the `dwpose_pose_detector` Model Manager entry cached.
    if let Some(path) =
        crate::downloads::resolve_hf_component_file(settings, DWPOSE_REPO, DWPOSE_REVISION, file)
    {
        return Ok(Some(path));
    }
    // Legacy pre-migration roots, read-only (AC10) — nothing writes to either any more.
    let app_cache = settings
        .data_dir
        .join(DWPOSE_LEGACY_DATA_DIR_ROOT)
        .join(file);
    if app_cache.exists() {
        return Ok(Some(app_cache));
    }
    // rtmlib's own checkpoint cache, present on dev machines / prior Python installs. Keyed off
    // `$HOME` rather than `data_dir`, so it survives an app-data move and is not reclaimable
    // through the Models screen.
    if let Some(home) = std::env::var_os("HOME") {
        let rtmlib = PathBuf::from(home)
            .join(DWPOSE_LEGACY_RTMLIB_ROOT)
            .join(file);
        if rtmlib.exists() {
            return Ok(Some(rtmlib));
        }
    }
    Ok(None)
}

/// [`resolve_dwpose_graph`], or an actionable install error naming the model and the env pin.
fn require_dwpose_graph(env_key: &str, file: &str, settings: &Settings) -> WorkerResult<PathBuf> {
    resolve_dwpose_graph(env_key, file, settings)?.ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "The DWPose pose detector is not installed ({DWPOSE_REPO}/{file}). Install \"DWPose \
             Pose Detector\" from the Model Manager, or point {env_key} at a local {file}."
        ))
    })
}

// ---------------------------------------------------------------------------
// source resolution (mirrors pose_adapters `_normalize_sources` / `_resolve_source_path`)
// ---------------------------------------------------------------------------

struct PoseSource {
    asset_id: Option<String>,
    display_name: Option<String>,
    temp: bool,
    path: Option<PathBuf>,
}

fn normalize_sources(payload: &JsonObject) -> Vec<JsonObject> {
    if let Some(arr) = payload.get("sources").and_then(Value::as_array) {
        return arr.iter().filter_map(|s| s.as_object().cloned()).collect();
    }
    if let Some(path) = optional_payload_string(payload, "path") {
        let mut single = JsonObject::new();
        single.insert("path".to_owned(), Value::String(path.to_owned()));
        if let Some(id) = optional_payload_string(payload, "sourceAssetId") {
            single.insert("assetId".to_owned(), Value::String(id.to_owned()));
        }
        return vec![single];
    }
    Vec::new()
}

fn resolve_source(
    src: &JsonObject,
    store: &ProjectStore,
    settings: &Settings,
    project_id: Option<&str>,
    project_path: Option<&Path>,
) -> WorkerResult<PoseSource> {
    let asset_id = src
        .get("assetId")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let display_name = src
        .get("displayName")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let temp = src.get("temp").and_then(Value::as_bool).unwrap_or(false);
    let raw = src.get("path").and_then(Value::as_str);

    // Staged upload path, confined to the same pose-uploads cache cleaned up below.
    if let Some(raw) = raw {
        let abs = Path::new(raw);
        if abs.is_absolute() {
            let path = normalize_app_managed_cache_path(
                settings,
                raw,
                POSE_UPLOADS_CACHE_DIR,
                "pose sourcePath",
            )?;
            if path.exists() {
                return Ok(PoseSource {
                    asset_id,
                    display_name,
                    temp,
                    path: Some(path),
                });
            }
            return Ok(PoseSource {
                asset_id,
                display_name,
                temp,
                path: None,
            });
        }
    }
    // asset id resolved against the originating project (Create tab sends ids)
    if let (Some(id), Some(pid), Some(proj)) = (&asset_id, project_id, project_path) {
        if let Some(asset) = resolve_asset_media(store, pid, id, proj) {
            return Ok(PoseSource {
                asset_id,
                display_name: display_name.or(asset.display_name),
                temp,
                path: Some(asset.path),
            });
        }
    }
    // project-relative path, confined to the project tree: reject any `..`/root/prefix
    // component so a crafted `../../<anything>` can't escape `proj` (sc-8875). The raw
    // cwd-relative fallback is dropped for the same reason — an unresolved relative path
    // would otherwise reach `decode_image_any` resolved against the worker's cwd.
    if let (Some(raw), Some(proj)) = (raw, project_path) {
        if let Some(candidate) = join_project_relative(proj, raw) {
            if candidate.exists() {
                return Ok(PoseSource {
                    asset_id,
                    display_name,
                    temp,
                    path: Some(candidate),
                });
            }
        }
    }
    Ok(PoseSource {
        asset_id,
        display_name,
        temp,
        path: None,
    })
}

/// Join a project-relative source path under `project_path`, accepting only
/// `Component::Normal` segments so a payload can never traverse out of the project
/// tree with `..`, an absolute root, or a drive/UNC prefix (sc-8875). Returns `None`
/// if any component is non-`Normal` (the caller then treats the source as unresolved).
fn join_project_relative(project_path: &Path, raw: &str) -> Option<PathBuf> {
    sceneworks_core::safe_path::confine_under(
        project_path,
        raw,
        sceneworks_core::safe_path::Confinement::SymlinkSafe,
    )
    .ok()
}

// ---------------------------------------------------------------------------
// job handler
// ---------------------------------------------------------------------------

pub(crate) async fn run_pose_detect_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Preparing DWPose detector.",
            None,
            None,
            None,
        ),
    )
    .await?;

    let payload = &job.payload;
    let sources = normalize_sources(payload);
    if sources.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "No source images supplied for pose detection.".to_owned(),
        ));
    }
    let min_conf = payload
        .get("minConf")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .unwrap_or(DEFAULT_POSE_MIN_CONF);

    let project_id = payload
        .get("projectId")
        .and_then(Value::as_str)
        .or(job.project_id.as_deref());
    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project_path = project_id
        .and_then(|pid| store.get_project(pid).ok())
        .map(|p| PathBuf::from(p.path));

    let resolved: Vec<PoseSource> = sources
        .iter()
        .map(|s| resolve_source(s, &store, settings, project_id, project_path.as_deref()))
        .collect::<WorkerResult<_>>()?;

    // Snapshot the staged File-Upload temp paths up front so cleanup runs on EVERY exit
    // path — success, cancel, or an intervening error (sc-8912). The old code only cleaned
    // up after the render finished, so a cancel/detect/render failure leaked the staged
    // uploads. `resolved` is later moved into the render task, so this list (cheap: just
    // the temp paths) is what the deferred cleanup keys off, independent of `resolved`.
    let temp_upload_paths: Vec<PathBuf> = resolved
        .iter()
        .filter(|s| s.temp)
        .filter_map(|s| s.path.clone())
        .collect();

    // Run the fallible body, then clean up the temp uploads regardless of outcome before
    // propagating (defer-style). Cleanup is confined to the pose-uploads cache inside
    // `cleanup_temp_uploads`, so a project asset resolved by id is never touched.
    let result = run_pose_detect_inner(api, settings, job, resolved, min_conf).await;
    cleanup_temp_uploads(settings, &temp_upload_paths).await;
    result
}

#[allow(clippy::too_many_arguments)]
async fn run_pose_detect_inner(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    resolved: Vec<PoseSource>,
    min_conf: f64,
) -> WorkerResult<()> {
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            // `Preparing`, not `Downloading`: since sc-17634 this only RESOLVES an
            // already-installed snapshot, so reporting a download stage would tell the UI a
            // transfer is happening that cannot occur.
            ProgressStage::Preparing,
            0.1,
            crate::load_progress::PREPARING_MODEL,
            None,
            None,
            None,
        ),
    )
    .await?;
    let (det_path, pose_path) = require_dwpose_weights(settings)?;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.3,
            "Detecting poses.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let image_paths: Vec<Option<PathBuf>> = resolved.iter().map(|s| s.path.clone()).collect();
    // Keep the worker heartbeat alive across the blocking batch detect (cold RTMW/onnx load +
    // multi-image inference can run long) so it never trips the API's 90s stale-sweep (sc-8390).
    // The batch loop is worker code, so it takes a REAL `CancelFlag` (sc-9123): the keepalive's
    // cancel poll trips it and `detect_batch` bails between images with the typed `Canceled`
    // instead of running the whole multi-image batch to its natural end. The same flag covers the
    // skeleton render below — a cancel that lands late in the detect stage still stops the job at
    // the next checkpoint, whichever stage that is.
    let cancel = gen_core::CancelFlag::new();
    let detect_cancel = cancel.clone();
    let (raw, device) = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel.clone()),
        POSE_CANCEL_MESSAGE,
        "pose detection task",
        crate::no_cancel_ack(),
        crate::load_progress::spawn_blocking(move || {
            detect_batch(det_path, pose_path, image_paths, &detect_cancel)
        }),
    )
    .await?;

    let out_dir = settings
        .data_dir
        .join("cache")
        .join("pose_detect")
        .join(&job.id);
    tokio::fs::create_dir_all(&out_dir).await?;

    // The post-inference render loop is itself CPU-heavy blocking work: `draw_wholebody` rasters a
    // `max(w,h)²` RGB canvas per person (~108 MB at 6000²) and `skeleton.save()` synchronously PNG-
    // encodes it. Running that inline on the async fn (no heartbeat arm live) blocks a tokio runtime
    // thread and grows the silent window toward the 90s stale-sweep on large multi-person high-res
    // batches — the same failure class sc-8390 fixed for the inference half. Fold it into a
    // `spawn_blocking` under `run_blocking_with_heartbeat` so it's both off-thread AND heartbeat-
    // covered (sc-8848). `resolved` is moved into the render task (it owns the source paths + asset
    // ids for the output records); temp-upload cleanup keys off the snapshot the caller took before
    // this move (sc-8912), so it doesn't need handing back. The render loop is worker code too, so it
    // shares the detect stage's `CancelFlag` (sc-9123) and bails between sources AND between
    // per-person rasters (each is a `max(w,h)²` canvas + PNG encode — the expensive unit) with the
    // typed `Canceled`; any skeleton PNGs already written sit in the job-scoped cache dir and are inert.
    let render_cancel = cancel.clone();
    let out_sources = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        POSE_CANCEL_MESSAGE,
        "pose skeleton render task",
        crate::no_cancel_ack(),
        crate::load_progress::spawn_blocking(move || {
            let mut out_sources: Vec<Value> = Vec::new();
            for (src, raw_src) in resolved.iter().zip(raw) {
                bail_if_canceled(&render_cancel)?;
                let stem = src
                    .path
                    .as_ref()
                    .and_then(|p| p.file_stem())
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "source".to_owned());
                let source_path_str = src.path.as_ref().map(|p| p.to_string_lossy().into_owned());
                let Some(raw_src) = raw_src else {
                    out_sources.push(json!({
                        "sourceAssetId": src.asset_id,
                        "sourcePath": source_path_str,
                        "error": "unreadable image",
                        "poses": [],
                    }));
                    continue;
                };
                let (w, h) = (raw_src.width as f32, raw_src.height as f32);

                // convert + squareify + order largest-person-area first
                let mut ordered: Vec<(f32, PoseRecord, Option<[f32; 4]>)> = raw_src
                    .people
                    .iter()
                    .map(|kps| {
                        let rec = squareify(&wholebody_to_openpose(kps, w, h)?, w, h);
                        let bb = bbox(
                            &[&rec.keypoints, &rec.hands[0], &rec.hands[1], &rec.face],
                            min_conf,
                        );
                        let area = bb.map_or(0.0, |b| (b[2] - b[0]) * (b[3] - b[1]));
                        Ok((area, rec, bb))
                    })
                    .collect::<WorkerResult<Vec<_>>>()?;
                ordered
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                let side = raw_src.width.max(raw_src.height);
                let stick = body_stickwidth(side, side);
                let mut poses: Vec<Value> = Vec::new();
                for (person_index, (_area, rec, bb)) in ordered.iter().enumerate() {
                    bail_if_canceled(&render_cancel)?;
                    let body_t = thresholded(&rec.keypoints, min_conf);
                    let hands_t = [
                        thresholded(&rec.hands[0], min_conf),
                        thresholded(&rec.hands[1], min_conf),
                    ];
                    let face_t = thresholded(&rec.face, min_conf);
                    let skeleton =
                        draw_wholebody(side, side, &body_t, Some(&hands_t), Some(&face_t), stick);
                    let preview = out_dir.join(format!("{stem}_p{person_index}_skel.png"));
                    // A failed preview encode/write is an output-side fault, not a bad user
                    // payload — classify it as `Engine` so it isn't mislabeled as user error
                    // (sc-8952). `ImageError` is not an `io::Error`, so `Io` doesn't fit.
                    skeleton.save(&preview).map_err(|e| {
                        WorkerError::Engine(format!("pose preview write: {e}"))
                    })?;

                    poses.push(json!({
                        "personIndex": person_index,
                        "bbox": bb,
                        "facing": facing(&rec.keypoints, min_conf),
                        "meanConf": {
                            "body": mean_conf(&rec.keypoints),
                            "hands": ((mean_conf(&rec.hands[0]) + mean_conf(&rec.hands[1])) / 2.0 * 1000.0).round() / 1000.0,
                            "face": mean_conf(&rec.face),
                        },
                        "keypoints": record_to_json(&rec.keypoints),
                        "hands": [record_to_json(&rec.hands[0]), record_to_json(&rec.hands[1])],
                        "face": record_to_json(&rec.face),
                        "skeletonPreview": preview.to_string_lossy(),
                    }));
                }

                out_sources.push(json!({
                    "sourceAssetId": src.asset_id,
                    "sourcePath": source_path_str,
                    "displayName": src.display_name.clone().unwrap_or_else(|| stem.clone()),
                    "sourceWidth": raw_src.width,
                    "sourceHeight": raw_src.height,
                    "sourceAspect": (w / h * 10000.0).round() / 10000.0,
                    "poses": poses,
                }));
            }
            Ok(out_sources)
        }),
    )
    .await?;

    let mut result = JsonObject::new();
    result.insert("sources".to_owned(), Value::Array(out_sources));
    result.insert(
        "detector".to_owned(),
        json!({"id": DETECTOR_ID, "device": device, "minConf": min_conf}),
    );
    result.insert("poseDetectionActive".to_owned(), Value::Bool(true));

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Pose detection complete.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

/// Delete the staged File-Upload temp sources, confined to the pose-uploads cache so a
/// project asset resolved by id is never removed. Takes the pre-computed temp paths (the
/// caller snapshots them before `resolved` is moved into the render task) so this runs on
/// every exit path — success OR error (sc-8912). Best-effort; a missing/already-removed
/// file is a no-op.
async fn cleanup_temp_uploads(settings: &Settings, temp_paths: &[PathBuf]) {
    if temp_paths.is_empty() {
        return;
    }
    let uploads_root = settings.data_dir.join("cache").join(POSE_UPLOADS_CACHE_DIR);
    let Ok(uploads_root) = uploads_root.canonicalize() else {
        return;
    };
    for path in temp_paths {
        if let Ok(resolved) = path.canonicalize() {
            if resolved.starts_with(&uploads_root) {
                let _ = tokio::fs::remove_file(&resolved).await;
            }
        }
    }
}

#[cfg(test)]
mod tests;
