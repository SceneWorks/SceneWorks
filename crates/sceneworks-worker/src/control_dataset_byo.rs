//! Bring-your-own-dataset ingest adapter — the second dataset-input path (Training Studio A3,
//! sc-10171, epic 10159). Where A2 GENERATES a dataset from a raw image folder, A3 maps a dataset
//! the user already PROVIDES into the same on-disk control-dataset layout
//! ([`crate::control_dataset_prep`]'s `manifest.jsonl` + `train/` triples), SKIPPING whatever the
//! source already supplies.
//!
//! Two structurally-different provided shapes:
//! - **Prepared** — the conditioning is ALREADY rendered (target + condition image [+ caption]).
//!   Ref: `raulc0399/open_pose_controlnet`. Ingested per-item after **condition-convention
//!   validation** ([`validate_pose_convention`]) routes it: matches → as-is; palette/aspect
//!   difference → normalize; structural (component-set) mismatch → regenerate from the target via
//!   the A1 preprocessor (a prepared dataset has only rendered pixels, no keypoints, so a wrong
//!   component set — e.g. raulc0399's body+hands under 8459's LOCKED body-only Config-1 — cannot be
//!   "fixed", only re-rendered).
//! - **Annotated** — keypoints/labels are provided but the conditioning is NOT rendered. Ref: COCO
//!   `person_keypoints` + `captions` JSON. The OpenPose-18 skeleton is rendered from the GROUND-TRUTH
//!   COCO-17 keypoints (no detection pass), captions come from the provided JSON.
//!
//! Shared with A2: the square-canonical letterbox, the `write_control_item` / `write_manifest` tail,
//! and (for the regenerate route) the A1 `preprocessor_for` re-render. Both ingest paths therefore
//! emit an identical dataset the Krea control trainer (B2) consumes.
//!
//! **Deferred (filed follow-ups), NOT here:** network fetch/unpack (HF-dataset / COCO-zip download)
//! — A3 ingests an already-extracted directory; and the in-memory Parquet reader for raulc0399
//! (which takes the regenerate route anyway under Config-1). Captioning falls back to A2's injected
//! caption input when a provided dataset lacks captions.
//!
//! Platform: the annotated-COCO render is cross-platform (GT keypoints → `draw_wholebody`, no
//! detector); prepared as-is/normalize ingest is cross-platform; only the regenerate route needs the
//! backend-gated `preprocessor_for` (pose/depth), erroring on a candle-disabled off-Mac build.

use std::path::{Path, PathBuf};

use gen_core::ControlKind;
use image::RgbImage;
use serde::Deserialize;

use crate::control_dataset_prep::{
    letterbox_to_square, write_control_item, write_manifest, PreppedItem,
};
use crate::control_preprocess::{
    control_kind_label, preprocessor_for, PoseComponents, PreprocessResources,
};
use crate::openpose_skeleton::{body_stickwidth, draw_wholebody, Keypoint};
use crate::{WorkerError, WorkerResult};

/// COCO-17 body keypoint → OpenPose-18 index `(op_idx, coco_idx)`. Identical to
/// `pose_jobs::COCO_TO_OPENPOSE` (COCO-WholeBody-133's first 17 ARE COCO-17), duplicated here so the
/// annotated-COCO render stays cross-platform (the pose_jobs detector module is backend-gated).
/// OpenPose-18 inserts neck(1) = midpoint of the two shoulders, computed separately.
const COCO17_TO_OPENPOSE: [(usize, usize); 17] = [
    (0, 0),
    (2, 6),
    (3, 8),
    (4, 10),
    (5, 5),
    (6, 7),
    (7, 9),
    (8, 12),
    (9, 14),
    (10, 16),
    (11, 11),
    (12, 13),
    (13, 15),
    (14, 2),
    (15, 1),
    (16, 4),
    (17, 3),
];

/// The declared convention of a pose condition — enough to decide whether a PROVIDED skeleton can be
/// used against a target backbone's inference preprocessor convention. Palette/topology beyond the
/// component set are assumed to be the standard OpenPose render (`openpose_palette`); a non-OpenPose
/// palette can't be recolored without the keypoints, so it routes to regenerate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PoseConditionConvention {
    /// Which keypoint groups the skeleton draws (body / body+hands / whole-body).
    pub(crate) components: PoseComponents,
    /// Whether the render uses the standard OpenPose limb/joint palette.
    pub(crate) openpose_palette: bool,
}

/// The route condition-convention validation picks for a provided item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum IngestRoute {
    /// Provided condition matches the target convention → ingest verbatim (letterboxed for aspect).
    AsIs,
    /// Fixable difference (currently: aspect only, handled by the shared square letterbox) → ingest
    /// with the listed normalizations + a warning.
    Normalize(Vec<String>),
    /// Structural mismatch (wrong component set, or a palette that can't be recolored without
    /// keypoints) → discard the provided condition and REGENERATE from the target. Carries the
    /// human-readable reason.
    Regenerate(String),
}

/// Validate a provided pose condition against the target backbone's required convention.
///
/// A component-set difference is structural — a prepared dataset has no keypoints, so extra
/// components (raulc0399 body+hands vs body-only Config-1) can't be subtracted nor missing ones
/// added; it routes to regenerate. A non-OpenPose palette likewise can't be recolored without
/// keypoints → regenerate. Otherwise the only possible difference is aspect, which the shared square
/// letterbox already fixes → `Normalize` (or `AsIs` when nothing differs).
pub(crate) fn validate_pose_convention(
    provided: PoseConditionConvention,
    required: PoseConditionConvention,
) -> IngestRoute {
    if provided.components != required.components {
        return IngestRoute::Regenerate(format!(
            "component-set mismatch: provided {:?}, target requires {:?} \
             (a rendered skeleton can't be re-componented without keypoints)",
            provided.components, required.components
        ));
    }
    if provided.openpose_palette != required.openpose_palette {
        return IngestRoute::Regenerate(
            "non-OpenPose palette can't be recolored without keypoints; regenerating".to_owned(),
        );
    }
    // Same components + palette: the only remaining difference is aspect, reconciled by the square
    // letterbox that every ingest path applies.
    IngestRoute::AsIs
}

/// One provided (target, already-rendered condition) pair, with the caption and the declared
/// convention of the condition.
pub(crate) struct PreparedPair {
    pub(crate) target_path: PathBuf,
    pub(crate) condition_path: PathBuf,
    pub(crate) caption: String,
    /// The convention the SOURCE rendered its condition in (declared by the source / user).
    pub(crate) provided_convention: PoseConditionConvention,
}

/// How to ingest a prepared pose dataset.
pub(crate) struct PreparedPoseConfig {
    /// The target backbone's required pose convention.
    pub(crate) required: PoseConditionConvention,
    /// Resources for the regenerate route (DWPose weights + render components).
    pub(crate) regenerate_resources: PreprocessResources,
}

/// Outcome of an ingest run.
pub(crate) struct ByoReport {
    pub(crate) items: Vec<PreppedItem>,
    /// `(item id, warning)` for normalized/regenerated/route decisions surfaced to the studio.
    pub(crate) warnings: Vec<(String, String)>,
    /// Items ingested verbatim.
    pub(crate) as_is: usize,
    /// Items whose condition was regenerated from the target.
    pub(crate) regenerated: usize,
    pub(crate) manifest_path: PathBuf,
}

/// Load an image as RGB, mapping a decode failure to an engine error.
fn load_rgb(path: &Path) -> WorkerResult<RgbImage> {
    Ok(crate::image_decode::decode_image_any(path)
        .map_err(|e| WorkerError::InvalidPayload(format!("open {}: {e}", path.display())))?
        .to_rgb8())
}

/// Ingest a PREPARED pose dataset (target + already-rendered skeleton pairs). Each pair is routed by
/// [`validate_pose_convention`]: matching conditions are letterboxed and written verbatim; a
/// structural/palette mismatch discards the provided skeleton and re-renders it from the target via
/// the A1 pose preprocessor. Writes the shared `manifest.jsonl`.
///
/// The regenerate route builds the pose preprocessor once (only if any pair needs it), so a
/// fully-matching dataset needs no detector weights. Blocking; call under `spawn_blocking`.
pub(crate) fn ingest_prepared_pose_pairs(
    pairs: &[PreparedPair],
    config: &PreparedPoseConfig,
    out_dir: &Path,
) -> WorkerResult<ByoReport> {
    let label = control_kind_label(&ControlKind::Pose);
    std::fs::create_dir_all(out_dir.join("train"))?;

    let mut items: Vec<PreppedItem> = Vec::new();
    let mut warnings: Vec<(String, String)> = Vec::new();
    let mut as_is = 0usize;
    let mut regenerated = 0usize;
    // The pose preprocessor for the regenerate route is built lazily on first need.
    let mut regenerator: Option<Box<dyn crate::control_preprocess::ControlPreprocessor>> = None;

    for (index, pair) in pairs.iter().enumerate() {
        let id = format!("{:06}", index + 1);
        let route = validate_pose_convention(pair.provided_convention, config.required);

        let target_square = letterbox_to_square(&load_rgb(&pair.target_path)?);
        let control = match &route {
            IngestRoute::AsIs => {
                as_is += 1;
                letterbox_to_square(&load_rgb(&pair.condition_path)?)
            }
            IngestRoute::Normalize(fixes) => {
                as_is += 1;
                warnings.push((id.clone(), format!("normalized: {}", fixes.join(", "))));
                letterbox_to_square(&load_rgb(&pair.condition_path)?)
            }
            IngestRoute::Regenerate(reason) => {
                regenerated += 1;
                warnings.push((id.clone(), format!("regenerated ({reason})")));
                if regenerator.is_none() {
                    regenerator = Some(preprocessor_for(
                        &ControlKind::Pose,
                        &config.regenerate_resources,
                    )?);
                }
                regenerator
                    .as_ref()
                    .expect("regenerator built")
                    .preprocess(&target_square)?
            }
        };

        items.push(write_control_item(
            out_dir,
            &id,
            &target_square,
            &control,
            &pair.caption,
            &label,
        )?);
    }

    let manifest_path = write_manifest(out_dir, &items, &label)?;
    Ok(ByoReport {
        items,
        warnings,
        as_is,
        regenerated,
        manifest_path,
    })
}

// --- Annotated (COCO person_keypoints + captions) --------------------------------------------

/// A COCO images/annotations JSON envelope (only the fields we read).
#[derive(Deserialize)]
struct CocoImages {
    images: Vec<CocoImage>,
}

#[derive(Deserialize)]
struct CocoImage {
    id: i64,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Deserialize)]
struct CocoKeypointFile {
    annotations: Vec<CocoKeypointAnn>,
}

#[derive(Deserialize)]
struct CocoKeypointAnn {
    image_id: i64,
    #[serde(default)]
    keypoints: Vec<f32>,
    #[serde(default)]
    num_keypoints: u32,
    #[serde(default)]
    area: f32,
}

#[derive(Deserialize)]
struct CocoCaptionFile {
    annotations: Vec<CocoCaptionAnn>,
}

#[derive(Deserialize)]
struct CocoCaptionAnn {
    image_id: i64,
    caption: String,
}

/// A COCO annotated source on disk: an images directory plus the person-keypoints and captions JSON.
pub(crate) struct CocoAnnotatedSource {
    pub(crate) images_dir: PathBuf,
    pub(crate) keypoints_json: PathBuf,
    pub(crate) captions_json: PathBuf,
    /// Render component set (8459 Config-1 = [`PoseComponents::Body`]).
    pub(crate) components: PoseComponents,
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> WorkerResult<T> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text)
        .map_err(|e| WorkerError::InvalidPayload(format!("parse {}: {e}", path.display())))
}

/// Convert one person's COCO-17 keypoints (`[x,y,v]×17`, original px) into thresholded OpenPose-18
/// render points, offset by `(ox, oy)` px into a `side`×`side` square and **normalized to [0,1]** —
/// the coordinate space `draw_wholebody` expects (its `placed`/`square_fit` maps [0,1] back to
/// canvas px). This mirrors `pose_jobs::squareify` (`(px + offset) / side`). A point with visibility
/// flag `v == 0` (unlabeled) becomes `None`.
fn coco17_to_openpose_render(keypoints: &[f32], ox: f32, oy: f32, side: f32) -> Vec<Keypoint> {
    let pt = |i: usize| -> Keypoint {
        let (x, y, v) = (keypoints[i * 3], keypoints[i * 3 + 1], keypoints[i * 3 + 2]);
        if v > 0.0 {
            Some(((x + ox) / side, (y + oy) / side))
        } else {
            None
        }
    };
    let mut body: Vec<Keypoint> = vec![None; 18];
    for (op, coco) in COCO17_TO_OPENPOSE {
        body[op] = pt(coco);
    }
    // Neck(1) = midpoint of the shoulders (COCO 5 = left, 6 = right) when both are labeled.
    let (ls, rs) = (pt(5), pt(6));
    body[1] = match (ls, rs) {
        (Some((lx, ly)), Some((rx, ry))) => Some(((lx + rx) / 2.0, (ly + ry) / 2.0)),
        _ => None,
    };
    body
}

/// Ingest a COCO ANNOTATED source: render the OpenPose-18 skeleton from ground-truth keypoints (no
/// detection), pair with the provided caption, and write the shared dataset. Images with no person
/// keypoints (or no caption) are skipped. The largest-area person per image is rendered.
///
/// Cross-platform (GT keypoints → `draw_wholebody`); no detector, no backend gate. Blocking; call
/// under `spawn_blocking`. `components` beyond `Body` is accepted but only the body-18 GT is
/// available from COCO person_keypoints, so hands/face render empty.
pub(crate) fn ingest_coco_annotated(
    source: &CocoAnnotatedSource,
    out_dir: &Path,
) -> WorkerResult<ByoReport> {
    let label = control_kind_label(&ControlKind::Pose);
    std::fs::create_dir_all(out_dir.join("train"))?;

    let images: CocoImages = read_json(&source.keypoints_json)?;
    let keypoints: CocoKeypointFile = read_json(&source.keypoints_json)?;
    let captions: CocoCaptionFile = read_json(&source.captions_json)?;

    // image_id → (file_name, w, h)
    let image_by_id: std::collections::HashMap<i64, CocoImage> =
        images.images.into_iter().map(|im| (im.id, im)).collect();
    // image_id → first caption
    let mut caption_by_id: std::collections::HashMap<i64, String> =
        std::collections::HashMap::new();
    for c in captions.annotations {
        caption_by_id.entry(c.image_id).or_insert(c.caption);
    }
    // image_id → largest-area person annotation with ≥1 keypoint.
    let mut best_ann: std::collections::HashMap<i64, CocoKeypointAnn> =
        std::collections::HashMap::new();
    for ann in keypoints.annotations {
        if ann.num_keypoints == 0 || ann.keypoints.len() < 51 {
            continue;
        }
        match best_ann.get(&ann.image_id) {
            Some(existing) if existing.area >= ann.area => {}
            _ => {
                best_ann.insert(ann.image_id, ann);
            }
        }
    }

    let mut items: Vec<PreppedItem> = Vec::new();
    let mut warnings: Vec<(String, String)> = Vec::new();

    // Deterministic order by image id.
    let mut ids: Vec<i64> = best_ann.keys().copied().collect();
    ids.sort_unstable();

    for (index, image_id) in ids.iter().enumerate() {
        let id = format!("{:06}", index + 1);
        let ann = &best_ann[image_id];
        let Some(image) = image_by_id.get(image_id) else {
            warnings.push((
                id,
                format!("image_id {image_id} has keypoints but no image entry"),
            ));
            continue;
        };
        let Some(caption) = caption_by_id.get(image_id) else {
            warnings.push((id, format!("image {} has no caption", image.file_name)));
            continue;
        };

        let target = match load_rgb(&source.images_dir.join(&image.file_name)) {
            Ok(img) => img,
            Err(_) => {
                warnings.push((id, format!("target image missing: {}", image.file_name)));
                continue;
            }
        };
        // COCO keypoints are in the ORIGINAL image px, which equals the decoded target dims; the
        // square canvas + offsets must match `letterbox_to_square(target)` so the skeleton aligns.
        let (tw, th) = (target.width(), target.height());
        let side = tw.max(th);
        let ox = (side - tw) as f32 / 2.0;
        let oy = (side - th) as f32 / 2.0;

        let body = coco17_to_openpose_render(&ann.keypoints, ox, oy, side as f32);
        let hands = source
            .components
            .wants_hands()
            .then(|| [Vec::<Keypoint>::new(), Vec::new()]);
        let face = source.components.wants_face().then(Vec::<Keypoint>::new);
        let stick = body_stickwidth(side, side);
        let skeleton = draw_wholebody(
            side,
            side,
            &body,
            hands.as_ref().map(|h| &h[..]),
            face.as_deref(),
            stick,
        );

        let target_square = letterbox_to_square(&target);
        items.push(write_control_item(
            out_dir,
            &id,
            &target_square,
            &skeleton,
            caption,
            &label,
        )?);
    }

    let manifest_path = write_manifest(out_dir, &items, &label)?;
    let as_is = items.len();
    Ok(ByoReport {
        items,
        warnings,
        as_is,
        regenerated: 0,
        manifest_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_only() -> PoseConditionConvention {
        PoseConditionConvention {
            components: PoseComponents::Body,
            openpose_palette: true,
        }
    }

    #[test]
    fn convention_routes_match_normalize_and_structural_mismatch() {
        // Same components + palette → as-is.
        assert_eq!(
            validate_pose_convention(body_only(), body_only()),
            IngestRoute::AsIs
        );
        // Extra components (body+hands provided, body-only required) → regenerate.
        let body_hands = PoseConditionConvention {
            components: PoseComponents::BodyHands,
            openpose_palette: true,
        };
        assert!(matches!(
            validate_pose_convention(body_hands, body_only()),
            IngestRoute::Regenerate(_)
        ));
        // Non-OpenPose palette → regenerate.
        let odd_palette = PoseConditionConvention {
            components: PoseComponents::Body,
            openpose_palette: false,
        };
        assert!(matches!(
            validate_pose_convention(odd_palette, body_only()),
            IngestRoute::Regenerate(_)
        ));
    }

    #[test]
    fn prepared_pairs_ingest_matching_conditions_as_is() {
        let dir = tempfile::tempdir().expect("tempdir");
        // A provided (target, skeleton) pair, both non-square to exercise the letterbox.
        let target = dir.path().join("t.png");
        let cond = dir.path().join("t.pose.png");
        RgbImage::from_pixel(300, 400, image::Rgb([90, 90, 90]))
            .save(&target)
            .unwrap();
        let mut skel = RgbImage::new(300, 400);
        skel.put_pixel(150, 200, image::Rgb([255, 0, 0]));
        skel.save(&cond).unwrap();

        let pairs = vec![PreparedPair {
            target_path: target,
            condition_path: cond,
            caption: "a person".to_owned(),
            provided_convention: body_only(),
        }];
        let out = dir.path().join("ds");
        let report = ingest_prepared_pose_pairs(
            &pairs,
            &PreparedPoseConfig {
                required: body_only(),
                regenerate_resources: PreprocessResources::new(),
            },
            &out,
        )
        .expect("ingest");

        assert_eq!(report.as_is, 1);
        assert_eq!(report.regenerated, 0);
        assert_eq!(report.items.len(), 1);
        // Target + condition written square and aligned.
        let t = image::open(out.join(&report.items[0].target_rel)).unwrap();
        let c = image::open(out.join(&report.items[0].control_rel)).unwrap();
        assert_eq!(t.width(), t.height());
        assert_eq!((t.width(), t.height()), (c.width(), c.height()));
        assert_eq!(report.items[0].control_rel, "train/000001.pose.png");
    }

    #[test]
    fn coco_annotated_renders_skeleton_from_ground_truth() {
        let dir = tempfile::tempdir().expect("tempdir");
        let images_dir = dir.path().join("images");
        std::fs::create_dir_all(&images_dir).unwrap();
        // One 128x96 target image.
        RgbImage::from_pixel(128, 96, image::Rgb([40, 40, 40]))
            .save(images_dir.join("a.jpg"))
            .unwrap();

        // Minimal COCO person_keypoints: nose + both shoulders + both hips labeled (v=2), rest 0.
        let mut kps = vec![0.0f32; 51];
        let set = |k: &mut Vec<f32>, i: usize, x: f32, y: f32| {
            k[i * 3] = x;
            k[i * 3 + 1] = y;
            k[i * 3 + 2] = 2.0;
        };
        set(&mut kps, 0, 64.0, 20.0); // nose
        set(&mut kps, 5, 48.0, 40.0); // left shoulder
        set(&mut kps, 6, 80.0, 40.0); // right shoulder
        set(&mut kps, 11, 52.0, 70.0); // left hip
        set(&mut kps, 12, 76.0, 70.0); // right hip
        let kps_json = serde_json::json!({
            "images": [{"id": 1, "file_name": "a.jpg", "width": 128, "height": 96}],
            "annotations": [{"image_id": 1, "keypoints": kps, "num_keypoints": 5, "area": 4000.0}]
        });
        let cap_json = serde_json::json!({
            "images": [{"id": 1, "file_name": "a.jpg"}],
            "annotations": [{"image_id": 1, "caption": "a standing person"}]
        });
        let kp_path = dir.path().join("person_keypoints.json");
        let cap_path = dir.path().join("captions.json");
        std::fs::write(&kp_path, kps_json.to_string()).unwrap();
        std::fs::write(&cap_path, cap_json.to_string()).unwrap();

        let report = ingest_coco_annotated(
            &CocoAnnotatedSource {
                images_dir,
                keypoints_json: kp_path,
                captions_json: cap_path,
                components: PoseComponents::Body,
            },
            &dir.path().join("ds"),
        )
        .expect("coco ingest");

        assert_eq!(report.items.len(), 1, "one annotated image rendered");
        assert_eq!(report.items[0].caption, "a standing person");
        let out = dir.path().join("ds");
        let skel = image::open(out.join(&report.items[0].control_rel))
            .unwrap()
            .to_rgb8();
        assert_eq!(skel.width(), skel.height(), "square canvas (max(w,h)=128)");
        assert_eq!(skel.width(), 128);
        // The skeleton is non-empty (limbs drawn between the labeled joints).
        assert!(
            skel.pixels().any(|p| p.0 != [0, 0, 0]),
            "skeleton must draw limbs from the GT keypoints"
        );
    }
}
