//! Folder-ingest control-dataset prep — the "create your own" GENERATE core (Training Studio A2,
//! sc-10161, epic 10159).
//!
//! Turns a set of raw target images into a trainer-ready **control dataset**: for each target it
//! renders the condition image with the A1 preprocessor registry ([`crate::control_preprocess`]),
//! pairs it with a caption, writes the `(target, control, caption)` triple to disk, and emits a
//! `manifest.jsonl` — the exact on-disk layout the Krea control trainer (B2, `krea_2_control`) and
//! its GPU smoke consume, and the format the bring-your-own-dataset adapter (A3, sc-10171) shares.
//!
//! **Backbone-agnostic.** This stage produces raw images; the trainer VAE-/text-encodes them
//! itself (the optional pre-encoded `x0/ctrl/cap` ControlSample cache is a per-backbone perf
//! optimization tracked separately, not emitted here). Because the ONLY thing that varies per
//! control type is the preprocessor, adding a control type is free here — it flows straight through
//! `preprocessor_for`.
//!
//! **Aspect: square-canonical (resolves 8459 S1's open native-vs-square sub-decision).** The A1
//! pose renderer draws the skeleton on a `max(w,h)` square canvas (`squareify`), so a native-aspect
//! target and its pose condition would not pixel-align. To make alignment automatic for EVERY
//! control kind, each target is first letterboxed to a centered square (pad the short axis, never
//! crop) and the preprocessor runs on that square — target and control then share identical
//! geometry. This matches the worker's existing square-canonical control placement and the proven
//! 5k COCO-pose spike (512² square). The trainer resizes the square to its configured resolution.
//!
//! **Captioning is an injected input, not run here.** Rich VLM captioning is the heaviest offline
//! pass and already ships as its own JoyCaption job (`caption_jobs.rs`); the studio job (B1) runs
//! it and feeds captions in, and A3 supplies provided captions. Keeping it out of the generate core
//! leaves this module a pure, cross-platform, unit-testable pipeline (canny needs no weights).
//!
//! Platform note: the canny generate path is cross-platform; pose/depth resolve through
//! `preprocessor_for` (neural-inference builds only) and the optional person filter needs the
//! backend-gated YOLO detector, so it errors on a candle-disabled off-Mac build.

use std::path::{Path, PathBuf};

use gen_core::ControlKind;
use image::{Rgb, RgbImage};
use serde_json::json;

use crate::control_preprocess::{control_kind_label, preprocessor_for, PreprocessResources};
use crate::{WorkerError, WorkerResult};

/// Default shorter-edge floor (px): a target smaller than this is skipped as too low-resolution to
/// yield a usable training pair. Deliberately permissive (warn-not-block philosophy — Dataset
/// Doctor's richer quality pass runs separately); only genuinely tiny images are dropped.
pub(crate) const DEFAULT_MIN_EDGE: u32 = 256;

/// Optional person-presence gate for pose/people corpora: drop targets with no detected person so a
/// pose ControlNet isn't trained on empty skeletons. Uses the worker's YOLO11 person detector
/// (`person_jobs::detect_people_blocking`), so it is only available on a neural-inference build.
pub(crate) struct PersonFilter {
    /// Resolved YOLO11 detector weights (see `person_jobs::ensure_detector_weights`).
    pub(crate) weights_path: PathBuf,
    /// Detection confidence floor.
    pub(crate) min_conf: f32,
}

/// One target image plus its caption. The caption is produced upstream (JoyCaption job / A3's
/// provided captions); this module does not run the VLM.
pub(crate) struct ControlPrepInput {
    pub(crate) target_path: PathBuf,
    pub(crate) caption: String,
}

/// How to generate the dataset.
pub(crate) struct ControlPrepConfig {
    /// Control type to render (pose / canny / depth). Resolved via [`preprocessor_for`].
    pub(crate) kind: ControlKind,
    /// Weights / options the preprocessor needs (A1 resource bundle).
    pub(crate) resources: PreprocessResources,
    /// Shorter-edge floor; targets below it are skipped ([`DEFAULT_MIN_EDGE`]).
    pub(crate) min_edge: u32,
    /// Optional person-presence gate (default `None`).
    pub(crate) person_filter: Option<PersonFilter>,
}

/// A written dataset row.
pub(crate) struct PreppedItem {
    pub(crate) id: String,
    /// Target image path, relative to the dataset root.
    pub(crate) target_rel: String,
    /// Control (condition) image path, relative to the dataset root.
    pub(crate) control_rel: String,
    pub(crate) caption: String,
}

/// Why a target was dropped (surfaced in the report so the studio can show a per-image tally rather
/// than silently shrinking the corpus).
#[derive(Debug)]
pub(crate) enum SkipReason {
    /// The image could not be decoded.
    Undecodable(String),
    /// Shorter edge below [`ControlPrepConfig::min_edge`].
    TooSmall { width: u32, height: u32 },
    /// Person filter enabled and no person was detected.
    NoPersonDetected,
    /// The preprocessor failed to render a condition for this image.
    PreprocessFailed(String),
}

/// Outcome of a prep run.
pub(crate) struct PrepReport {
    /// Successfully written `(target, control, caption)` rows.
    pub(crate) items: Vec<PreppedItem>,
    /// Dropped inputs with the reason (warn-not-block).
    pub(crate) skipped: Vec<(PathBuf, SkipReason)>,
    /// Path of the written `manifest.jsonl`.
    pub(crate) manifest_path: PathBuf,
}

/// Letterbox `img` into a centered `max(w,h)` black square (pad the short axis, never crop). Mirrors
/// `pose_jobs::squareify`'s centering so a square target aligns with the square-rendered skeleton.
fn letterbox_to_square(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    if w == h {
        return img.clone();
    }
    let side = w.max(h);
    let mut out = RgbImage::from_pixel(side, side, Rgb([0, 0, 0]));
    let ox = i64::from((side - w) / 2);
    let oy = i64::from((side - h) / 2);
    image::imageops::overlay(&mut out, img, ox, oy);
    out
}

/// Whether `path` contains a detected person (neural-inference build).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn target_has_person(path: &Path, filter: &PersonFilter) -> WorkerResult<bool> {
    let result = crate::person_jobs::detect_people_blocking(
        filter.weights_path.clone(),
        path.to_path_buf(),
        filter.min_conf,
    )?;
    Ok(!result.detections.is_empty())
}

#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
fn target_has_person(_path: &Path, _filter: &PersonFilter) -> WorkerResult<bool> {
    Err(WorkerError::Engine(
        "person filter requires a neural-inference build (macOS or off-Mac backend-candle)"
            .to_owned(),
    ))
}

/// Save an RGB image as PNG, classifying an encode/write failure as an engine (output-side) fault
/// rather than a bad user payload.
fn save_png(img: &RgbImage, path: &Path) -> WorkerResult<()> {
    img.save(path)
        .map_err(|e| WorkerError::Engine(format!("control-prep write {}: {e}", path.display())))
}

/// Generate a control training dataset from `inputs`.
///
/// For each target: decode → resolution gate → optional person gate → letterbox to square → render
/// the condition via the A1 preprocessor → write `train/{id}.target.png`, `train/{id}.{kind}.png`,
/// `train/{id}.txt` → collect a manifest row. Finally writes `out_dir/manifest.jsonl`
/// (`{"id","caption","target","control","kind"}` per line — the canonical control key is `control`;
/// pose datasets from before this stage used `pose`, which readers accept as a fallback).
///
/// The preprocessor is built ONCE and reused across the corpus. Skipped inputs are collected in the
/// report, not fatal. Blocking (decode + preprocess + PNG encode per image) — call under
/// `spawn_blocking`; `on_progress(done, total)` is invoked before each item and once at the end.
pub(crate) fn prepare_control_dataset(
    inputs: &[ControlPrepInput],
    config: &ControlPrepConfig,
    out_dir: &Path,
    mut on_progress: impl FnMut(usize, usize),
) -> WorkerResult<PrepReport> {
    let label = control_kind_label(&config.kind);
    // Fail fast if the requested kind can't be served on this build / with these resources, before
    // creating any output.
    let preprocessor = preprocessor_for(&config.kind, &config.resources)?;

    let train_dir = out_dir.join("train");
    std::fs::create_dir_all(&train_dir)?;

    let total = inputs.len();
    let mut items: Vec<PreppedItem> = Vec::new();
    let mut skipped: Vec<(PathBuf, SkipReason)> = Vec::new();

    for (index, input) in inputs.iter().enumerate() {
        on_progress(index, total);
        let id = format!("{:06}", index + 1);

        let decoded = match crate::image_decode::decode_image_any(&input.target_path) {
            Ok(image) => image.to_rgb8(),
            Err(e) => {
                skipped.push((
                    input.target_path.clone(),
                    SkipReason::Undecodable(e.to_string()),
                ));
                continue;
            }
        };
        let (w, h) = (decoded.width(), decoded.height());
        if w.min(h) < config.min_edge {
            skipped.push((
                input.target_path.clone(),
                SkipReason::TooSmall {
                    width: w,
                    height: h,
                },
            ));
            continue;
        }

        if let Some(filter) = &config.person_filter {
            if !target_has_person(&input.target_path, filter)? {
                skipped.push((input.target_path.clone(), SkipReason::NoPersonDetected));
                continue;
            }
        }

        // Square-canonical: letterbox the target, then render the condition from the SAME square so
        // target and control share geometry regardless of control kind.
        let target_square = letterbox_to_square(&decoded);
        let control = match preprocessor.preprocess(&target_square) {
            Ok(control) => control,
            Err(e) => {
                skipped.push((
                    input.target_path.clone(),
                    SkipReason::PreprocessFailed(e.to_string()),
                ));
                continue;
            }
        };

        let target_rel = format!("train/{id}.target.png");
        let control_rel = format!("train/{id}.{label}.png");
        save_png(&target_square, &out_dir.join(&target_rel))?;
        save_png(&control, &out_dir.join(&control_rel))?;
        std::fs::write(train_dir.join(format!("{id}.txt")), &input.caption)?;

        items.push(PreppedItem {
            id,
            target_rel,
            control_rel,
            caption: input.caption.clone(),
        });
    }
    on_progress(total, total);

    let manifest_path = out_dir.join("manifest.jsonl");
    let mut manifest = String::new();
    for item in &items {
        let row = json!({
            "id": item.id,
            "caption": item.caption,
            "target": item.target_rel,
            "control": item.control_rel,
            "kind": label,
        });
        manifest.push_str(&row.to_string());
        manifest.push('\n');
    }
    std::fs::write(&manifest_path, manifest)?;

    Ok(PrepReport {
        items,
        skipped,
        manifest_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `w`×`h` image with a bright off-center block, decodable + edge-bearing for canny.
    fn sample_image(w: u32, h: u32) -> RgbImage {
        let mut img = RgbImage::from_pixel(w, h, Rgb([20, 20, 20]));
        for y in (h / 4)..(h / 2) {
            for x in (w / 4)..(w / 2) {
                img.put_pixel(x, y, Rgb([240, 240, 240]));
            }
        }
        img
    }

    fn canny_config() -> ControlPrepConfig {
        ControlPrepConfig {
            kind: ControlKind::Canny,
            resources: PreprocessResources::new(),
            min_edge: DEFAULT_MIN_EDGE,
            person_filter: None,
        }
    }

    #[test]
    fn generates_square_aligned_pairs_and_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        // A non-square target (300x400) exercises the letterbox-to-square path.
        let target = dir.path().join("shot.png");
        sample_image(300, 400).save(&target).expect("write target");

        let inputs = vec![ControlPrepInput {
            target_path: target,
            caption: "a person, standing".to_owned(),
        }];
        let out = dir.path().join("dataset");
        let mut ticks = Vec::new();
        let report =
            prepare_control_dataset(&inputs, &canny_config(), &out, |d, t| ticks.push((d, t)))
                .expect("prep");

        assert_eq!(report.items.len(), 1, "one pair written");
        assert!(report.skipped.is_empty(), "nothing skipped");
        assert_eq!(report.items[0].control_rel, "train/000001.canny.png");
        assert!(
            ticks.last() == Some(&(1, 1)),
            "final progress tick is (total,total)"
        );

        // Target + control both exist, are square, and share identical dimensions (aligned).
        let target_img = image::open(out.join(&report.items[0].target_rel)).expect("target png");
        let control_img = image::open(out.join(&report.items[0].control_rel)).expect("control png");
        assert_eq!(
            target_img.width(),
            target_img.height(),
            "target letterboxed square"
        );
        assert_eq!(
            (target_img.width(), target_img.height()),
            (control_img.width(), control_img.height()),
            "control aligns with target geometry"
        );
        assert_eq!(target_img.width(), 400, "square side = max(w,h)");

        // Caption sidecar written.
        assert_eq!(
            std::fs::read_to_string(out.join("train/000001.txt")).unwrap(),
            "a person, standing"
        );

        // Manifest row is the canonical control shape.
        let manifest = std::fs::read_to_string(&report.manifest_path).expect("manifest");
        let row: serde_json::Value = serde_json::from_str(manifest.trim()).expect("row json");
        assert_eq!(row["target"], "train/000001.target.png");
        assert_eq!(row["control"], "train/000001.canny.png");
        assert_eq!(row["kind"], "canny");
        assert_eq!(row["caption"], "a person, standing");
    }

    #[test]
    fn skips_undecodable_and_too_small_without_failing_the_run() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Undecodable: a .png that isn't an image.
        let bogus = dir.path().join("bogus.png");
        std::fs::write(&bogus, b"not an image").unwrap();
        // Too small: below DEFAULT_MIN_EDGE.
        let tiny = dir.path().join("tiny.png");
        sample_image(64, 64).save(&tiny).unwrap();
        // Good.
        let good = dir.path().join("good.png");
        sample_image(320, 320).save(&good).unwrap();

        let inputs = vec![
            ControlPrepInput {
                target_path: bogus.clone(),
                caption: "x".to_owned(),
            },
            ControlPrepInput {
                target_path: tiny.clone(),
                caption: "y".to_owned(),
            },
            ControlPrepInput {
                target_path: good,
                caption: "z".to_owned(),
            },
        ];
        let out = dir.path().join("ds");
        let report =
            prepare_control_dataset(&inputs, &canny_config(), &out, |_, _| {}).expect("prep");

        assert_eq!(report.items.len(), 1, "only the good image is written");
        assert_eq!(report.skipped.len(), 2);
        assert!(report
            .skipped
            .iter()
            .any(|(_, r)| matches!(r, SkipReason::Undecodable(_))));
        assert!(report
            .skipped
            .iter()
            .any(|(_, r)| matches!(r, SkipReason::TooSmall { .. })));
        // The good image is renumbered by input index, so its id reflects position (000003).
        assert_eq!(report.items[0].id, "000003");
    }

    #[test]
    fn letterbox_pads_short_axis_centered() {
        let wide = RgbImage::from_pixel(40, 10, Rgb([255, 255, 255]));
        let sq = letterbox_to_square(&wide);
        assert_eq!((sq.width(), sq.height()), (40, 40));
        // Top + bottom rows are padding (black); the middle band holds the original white.
        assert_eq!(sq.get_pixel(0, 0).0, [0, 0, 0], "corner is pad");
        assert_eq!(sq.get_pixel(20, 20).0, [255, 255, 255], "center is content");
    }
}
