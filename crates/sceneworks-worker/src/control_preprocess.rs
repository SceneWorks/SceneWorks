//! Control-type preprocessor registry (ControlNet Training Studio A1, sc-10160, epic 10159).
//!
//! Formalizes the "control type == preprocessor" axis: a control type differs from another ONLY
//! in the function that turns a target image into its condition image (pose skeleton, canny edge
//! map, depth map). Everything downstream — control-branch arch, training loop, flow-matching
//! loss, latent cache, convert, overlay register, inference injection — is identical across types.
//!
//! This module is the single home for that mapping, keyed by [`gen_core::ControlKind`], wrapping
//! the EXISTING worker preprocessors ([`crate::openpose_skeleton`] via [`crate::pose_jobs`],
//! [`crate::canny`], [`crate::depth`]) behind one [`ControlPreprocessor`] trait. The data-prep
//! pipeline (A2, folder-ingest), the bring-your-own-dataset adapter (A3, annotated-render +
//! convention validation), and the strict-control inference lanes all resolve their preprocessor
//! here, so the SAME code renders the condition at train-prep AND generation time — the
//! convention-match is then automatic (the pose lesson: train/infer skeletons must be the
//! identical render convention). Adding a control type later == registering a preprocessor here,
//! with no change to the training or inference core.
//!
//! **Registry v1 wraps exactly three preprocessors: pose / canny / depth** — the control types
//! that already ship an inference lane. New preprocessors (hed / lineart / mlsd / scribble) are
//! out of scope here; they join via [`gen_core::ControlKind::Other`] (or a promoted variant) when
//! Phase C builds them.
//!
//! Platform gating mirrors the wrapped modules: canny is pure CPU raster and builds everywhere;
//! depth and pose need neural inference (Depth-Anything V2 / DWPose onnx) and so exist only on
//! `macos` or an off-Mac `backend-candle` build. On a candle-disabled off-Mac box the registry
//! still compiles (and the canny path unit-tests) but resolving a depth/pose preprocessor returns
//! a typed "unavailable on this build" error rather than failing to compile.

use gen_core::ControlKind;
use image::RgbImage;

use crate::{WorkerError, WorkerResult};

/// Default keypoint-confidence floor for pose detection: points scoring below it are dropped from
/// the rendered skeleton. Owned here (cross-platform) so the gated `pose_jobs` module and this
/// registry share one value; mirrors the retired Python worker's `DEFAULT_POSE_MIN_CONF`.
pub(crate) const DEFAULT_POSE_MIN_CONF: f64 = 0.3;

/// A stable lowercase label for a [`ControlKind`] — telemetry, error messages, the `controlMode`
/// request field, and dataset/overlay metadata all key off this. Single source of truth (the
/// strict-control driver delegates here so the label can't drift between train-prep and inference).
pub(crate) fn control_kind_label(kind: &ControlKind) -> String {
    match kind {
        ControlKind::Pose => "pose".to_owned(),
        ControlKind::Canny => "canny".to_owned(),
        ControlKind::Depth => "depth".to_owned(),
        ControlKind::Other(name) => name.clone(),
    }
}

/// Which keypoint groups a pose preprocessor renders into the skeleton condition.
///
/// Pose control is the same COCO-18 body skeleton regardless of set; the set only toggles the
/// optional hands (21×2) and dense face (68) overlays. The Training-Studio default is
/// [`PoseComponents::Body`] to match epic 8459's LOCKED Config-1 (body-18 + head-direction
/// keypoints — nose/eyes/ears, which live in the body-18 set — with NO hands and NO dense face):
/// head DIRECTION is the priority axis and is body-18-encoded, while the dense face is an
/// expression carrier that would over-constrain, and hands are the hardest control and not a
/// priority. The richer sets exist so the same registry serves a hands/face inference lane later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum PoseComponents {
    /// Body-18 + head-direction only (8459 Config-1). No hands, no dense face. The studio default.
    #[default]
    Body,
    // The richer sets exist so the same registry serves a hands/face inference lane later (epic
    // 10159); the training studio renders `Body` (Config-1) only today, so they are not yet
    // constructed. Kept as intentional forward API rather than deleted.
    #[allow(dead_code)]
    /// Body-18 + 21×2 hands. No dense face.
    BodyHands,
    #[allow(dead_code)]
    /// Full whole-body: body-18 + hands + 68-pt face (matches the `pose_detect` job's render).
    WholeBody,
}

impl PoseComponents {
    /// Whether the 21×2 hand keypoints are drawn.
    pub(crate) fn wants_hands(self) -> bool {
        matches!(self, Self::BodyHands | Self::WholeBody)
    }

    /// Whether the dense 68-pt face keypoints are drawn.
    pub(crate) fn wants_face(self) -> bool {
        matches!(self, Self::WholeBody)
    }
}

/// A control-type preprocessor: turns a source RGB image into the condition image the control
/// branch consumes at train time and the strict-control lane injects at inference time.
///
/// Object-safe on purpose — the registry hands back a `Box<dyn ControlPreprocessor>` so a caller
/// (folder-ingest, byte-your-own-dataset ingest, an inference lane) builds it once from resolved
/// resources and reuses it across a whole dataset / batch.
pub(crate) trait ControlPreprocessor: Send + Sync {
    // `kind`/`label` are the inference-facing accessors of the shared preprocessor contract; the
    // training-studio caller drives only `preprocess` (it labels the dataset via `control_kind_label`
    // directly), so they are not yet called off-Mac. Kept as intentional contract surface.
    /// The control kind this preprocessor produces.
    #[allow(dead_code)]
    fn kind(&self) -> ControlKind;

    /// The stable lowercase label ([`control_kind_label`]) for this preprocessor's kind.
    #[allow(dead_code)]
    fn label(&self) -> String {
        control_kind_label(&self.kind())
    }

    /// Render the condition image for `image`. Output dimensions follow the underlying
    /// preprocessor (canny/depth preserve input dims; pose renders a `max(w,h)` square canvas).
    fn preprocess(&self, image: &RgbImage) -> WorkerResult<RgbImage>;
}

/// Canny edge-map preprocessor ([`ControlKind::Canny`]). Pure CPU raster — available everywhere.
pub(crate) struct CannyPreprocessor {
    low_threshold: f32,
    high_threshold: f32,
}

impl CannyPreprocessor {
    /// Build with the standard ControlNet canny thresholds
    /// ([`crate::canny::DEFAULT_LOW_THRESHOLD`] / [`crate::canny::DEFAULT_HIGH_THRESHOLD`]).
    pub(crate) fn new_default() -> Self {
        Self {
            low_threshold: crate::canny::DEFAULT_LOW_THRESHOLD,
            high_threshold: crate::canny::DEFAULT_HIGH_THRESHOLD,
        }
    }
}

impl ControlPreprocessor for CannyPreprocessor {
    fn kind(&self) -> ControlKind {
        ControlKind::Canny
    }

    fn preprocess(&self, image: &RgbImage) -> WorkerResult<RgbImage> {
        Ok(crate::canny::canny_control_image(
            image,
            self.low_threshold,
            self.high_threshold,
        ))
    }
}

/// Depth-map preprocessor ([`ControlKind::Depth`]). Runs the Depth-Anything V2 (Small) estimator,
/// so it exists only on the neural-inference builds (macOS MLX / off-Mac candle).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) struct DepthPreprocessor {
    /// Snapshot dir holding `model.safetensors` (what [`crate::depth::depth_control_image`] loads).
    weights_dir: std::path::PathBuf,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl DepthPreprocessor {
    pub(crate) fn new(weights_dir: std::path::PathBuf) -> Self {
        Self { weights_dir }
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl ControlPreprocessor for DepthPreprocessor {
    fn kind(&self) -> ControlKind {
        ControlKind::Depth
    }

    fn preprocess(&self, image: &RgbImage) -> WorkerResult<RgbImage> {
        crate::depth::depth_control_image(image, &self.weights_dir)
    }
}

/// Pose-skeleton preprocessor ([`ControlKind::Pose`]). Detects the dominant person with DWPose and
/// renders an OpenPose skeleton via [`crate::pose_jobs::detect_and_render_skeleton`], so it exists
/// only on the neural-inference builds (macOS CoreML / off-Mac candle CUDA onnx EP).
///
/// Unlike the strict-control inference pose path (which renders a caller-supplied skeleton from
/// request keypoints), this DETECTS keypoints from the source image — the direction folder-ingest
/// train-prep needs. Both render through the same `draw_wholebody`, so the conventions match.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) struct PosePreprocessor {
    det_path: std::path::PathBuf,
    pose_path: std::path::PathBuf,
    components: PoseComponents,
    min_conf: f64,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl PosePreprocessor {
    /// Build from the two resolved DWPose onnx weight paths (detector + pose model). `components`
    /// selects the rendered groups (studio default [`PoseComponents::Body`]); `min_conf` is the
    /// keypoint-confidence floor ([`crate::pose_jobs::DEFAULT_POSE_MIN_CONF`]).
    pub(crate) fn new(
        det_path: std::path::PathBuf,
        pose_path: std::path::PathBuf,
        components: PoseComponents,
        min_conf: f64,
    ) -> Self {
        Self {
            det_path,
            pose_path,
            components,
            min_conf,
        }
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl ControlPreprocessor for PosePreprocessor {
    fn kind(&self) -> ControlKind {
        ControlKind::Pose
    }

    fn preprocess(&self, image: &RgbImage) -> WorkerResult<RgbImage> {
        crate::pose_jobs::detect_and_render_skeleton(
            &self.det_path,
            &self.pose_path,
            image,
            self.components,
            self.min_conf,
        )
    }
}

/// Already-provisioned resources a preprocessor needs, handed to [`preprocessor_for`].
///
/// Weight download / snapshot resolution is deliberately NOT done here — the worker already owns
/// those async provisioning helpers (`ensure_weights` for DWPose, `ensure_depth_estimator_dir` for
/// Depth-Anything), and the data-prep pipeline (A2) calls them once then hands the resolved paths
/// in. Keeping the registry synchronous and resource-in keeps it a pure, unit-testable mapping and
/// free of job/ApiClient plumbing. Canny needs no resources.
#[derive(Clone, Debug)]
pub(crate) struct PreprocessResources {
    /// DWPose weights `(detector_onnx, pose_onnx)` for [`ControlKind::Pose`].
    pub(crate) dwpose: Option<(std::path::PathBuf, std::path::PathBuf)>,
    /// Depth-Anything V2 (Small) snapshot dir for [`ControlKind::Depth`].
    pub(crate) depth_weights_dir: Option<std::path::PathBuf>,
    /// Rendered pose component set (default [`PoseComponents::Body`] = 8459 Config-1).
    pub(crate) pose_components: PoseComponents,
    /// Keypoint-confidence floor for pose detection.
    pub(crate) pose_min_conf: f64,
}

impl Default for PreprocessResources {
    fn default() -> Self {
        Self::new()
    }
}

impl PreprocessResources {
    /// Resources with the studio pose defaults (Config-1 body-only render, [`DEFAULT_POSE_MIN_CONF`]
    /// confidence floor) and no weights resolved yet. Hand-written rather than `#[derive(Default)]`
    /// because the derived `f64::default()` would zero the confidence floor (draw every keypoint),
    /// not use the intended 0.3.
    pub(crate) fn new() -> Self {
        Self {
            dwpose: None,
            depth_weights_dir: None,
            pose_components: PoseComponents::Body,
            pose_min_conf: DEFAULT_POSE_MIN_CONF,
        }
    }
}

/// Resolve the [`ControlPreprocessor`] for `kind` from already-provisioned `resources`.
///
/// This is the registry entry point A2/A3/inference share. Errors are typed:
/// - a required resource is missing (e.g. pose without DWPose weights) → [`WorkerError::Engine`];
/// - `kind` has no registered preprocessor ([`ControlKind::Other`], or depth/pose on a build
///   without neural inference) → [`WorkerError::InvalidPayload`].
// `resources` is consumed only by the depth/pose arms, which are compiled out on a build with no
// neural inference (off-Mac, no candle) — where the parameter is genuinely unused.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(unused_variables)
)]
pub(crate) fn preprocessor_for(
    kind: &ControlKind,
    resources: &PreprocessResources,
) -> WorkerResult<Box<dyn ControlPreprocessor>> {
    match kind {
        ControlKind::Canny => Ok(Box::new(CannyPreprocessor::new_default())),
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        ControlKind::Depth => {
            let dir = resources.depth_weights_dir.clone().ok_or_else(|| {
                WorkerError::Engine(
                    "depth preprocessor requires a resolved Depth-Anything V2 weights dir \
                     (PreprocessResources.depth_weights_dir)"
                        .to_owned(),
                )
            })?;
            Ok(Box::new(DepthPreprocessor::new(dir)))
        }
        #[cfg(any(
            target_os = "macos",
            all(not(target_os = "macos"), feature = "backend-candle")
        ))]
        ControlKind::Pose => {
            let (det, pose) = resources.dwpose.clone().ok_or_else(|| {
                WorkerError::Engine(
                    "pose preprocessor requires resolved DWPose weights \
                     (PreprocessResources.dwpose = (detector_onnx, pose_onnx))"
                        .to_owned(),
                )
            })?;
            Ok(Box::new(PosePreprocessor::new(
                det,
                pose,
                resources.pose_components,
                resources.pose_min_conf,
            )))
        }
        other => Err(WorkerError::InvalidPayload(format!(
            "no control preprocessor registered for kind '{}' on this build",
            control_kind_label(other)
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_kind_labels_are_stable_lowercase() {
        assert_eq!(control_kind_label(&ControlKind::Pose), "pose");
        assert_eq!(control_kind_label(&ControlKind::Canny), "canny");
        assert_eq!(control_kind_label(&ControlKind::Depth), "depth");
        assert_eq!(
            control_kind_label(&ControlKind::Other("scribble".to_owned())),
            "scribble"
        );
    }

    #[test]
    fn pose_components_group_selection() {
        // Config-1 (studio default): body only.
        assert_eq!(PoseComponents::default(), PoseComponents::Body);
        assert!(!PoseComponents::Body.wants_hands());
        assert!(!PoseComponents::Body.wants_face());
        // +hands.
        assert!(PoseComponents::BodyHands.wants_hands());
        assert!(!PoseComponents::BodyHands.wants_face());
        // full whole-body.
        assert!(PoseComponents::WholeBody.wants_hands());
        assert!(PoseComponents::WholeBody.wants_face());
    }

    /// Canny is the cross-platform preprocessor: the registry resolves it with no resources and it
    /// produces the same grayscale-broadcast edge map the `ControlKind::Canny` path consumes.
    #[test]
    fn registry_resolves_canny_everywhere_and_it_renders() {
        let res = PreprocessResources::new();
        let pre = preprocessor_for(&ControlKind::Canny, &res).expect("canny resolves");
        assert_eq!(pre.kind(), ControlKind::Canny);
        assert_eq!(pre.label(), "canny");

        // A black field with a centered white square → canny fires on the four borders.
        let mut img = RgbImage::new(64, 64);
        for y in 20..44 {
            for x in 20..44 {
                img.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
        let out = pre.preprocess(&img).expect("canny render");
        assert_eq!(out.dimensions(), (64, 64), "canny preserves input dims");
        assert!(
            out.pixels().all(|p| p[0] == p[1] && p[1] == p[2]),
            "edge map is grayscale broadcast to RGB"
        );
        assert!(
            out.pixels().any(|p| p.0 == [255, 255, 255]),
            "canny must fire on the square's borders"
        );
    }

    #[test]
    fn registry_rejects_unregistered_other_kind() {
        let res = PreprocessResources::new();
        // `Box<dyn ControlPreprocessor>` isn't `Debug`, so match on the result rather than
        // `expect_err` (which needs the Ok type to be `Debug`).
        assert!(matches!(
            preprocessor_for(&ControlKind::Other("hed".to_owned()), &res),
            Err(WorkerError::InvalidPayload(_))
        ));
    }

    /// On the neural-inference builds, pose/depth without resolved weights fail with a typed
    /// resource error rather than panicking or silently rendering nothing.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn registry_reports_missing_pose_and_depth_weights() {
        let res = PreprocessResources::new();
        assert!(
            matches!(
                preprocessor_for(&ControlKind::Pose, &res),
                Err(WorkerError::Engine(_))
            ),
            "pose without DWPose weights must be a typed resource error"
        );
        assert!(
            matches!(
                preprocessor_for(&ControlKind::Depth, &res),
                Err(WorkerError::Engine(_))
            ),
            "depth without a weights dir must be a typed resource error"
        );
    }
}
