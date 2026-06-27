//! Depth-map preprocessor for the Fun-Controlnet-Union depth head (epic 8236, sc-8242).
//! A native-MLX model-agnostic preprocessor — sibling of [`crate::canny`] /
//! `openpose_skeleton`: an arbitrary input image → a depth control image for
//! `ControlKind::Depth`.
//!
//! Unlike canny/pose (pure raster, cross-platform), depth needs real neural inference,
//! so it is **macOS-gated** and runs the native-MLX [`mlx_gen_depth`] port of Depth
//! Anything V2 (DINOv2 ViT-S/14 + DPT). The model is the **Small** variant
//! (`depth-anything/Depth-Anything-V2-Small-hf`, apache-2.0, ungated) — favoring
//! speed/size for a preprocessing tier.
//!
//! Output contract: an [`image::RgbImage`], the same type the pose preprocessor's
//! `draw_wholebody` and the canny preprocessor return and the `ControlKind::Depth` path
//! consumes — a single-channel depth map min/max-normalized to `[0,255]` and broadcast
//! across the three RGB channels (near = bright, the standard ControlNet depth
//! convention), at the input image's dimensions.

/// The Hugging Face repo for the default depth estimator: Depth Anything V2 **Small**
/// (apache-2.0, ungated — ships standard `model.safetensors`; no re-host needed). The
/// `-hf` (transformers) mirror is the safetensors-bearing one (the base
/// `depth-anything/Depth-Anything-V2-Small` ships only a `.pth`).
pub const DEPTH_ANYTHING_V2_SMALL_REPO: &str = "depth-anything/Depth-Anything-V2-Small-hf";

/// The single weight file the estimator loads from its snapshot dir.
pub const DEPTH_ANYTHING_V2_FILE: &str = "model.safetensors";

/// Estimate a depth control image from an arbitrary RGB input, loading the Depth Anything V2
/// estimator from `weights_dir` (a directory containing `model.safetensors`).
///
/// `img` is the source RGB image; the returned [`image::RgbImage`] is the normalized
/// grayscale-broadcast depth map at the SAME dimensions, drop-in for the `ControlKind::Depth`
/// path (the sibling of `canny::canny_control_image_default`). macOS-only (MLX inference).
#[cfg(target_os = "macos")]
pub fn depth_control_image(
    img: &image::RgbImage,
    weights_dir: &std::path::Path,
) -> crate::WorkerResult<image::RgbImage> {
    use crate::WorkerError;

    let model = mlx_gen_depth::DepthAnythingV2::from_dir(weights_dir)
        .map_err(|error| WorkerError::Engine(format!("depth estimator load: {error}")))?;
    let (w, h) = (img.width(), img.height());
    let control = model
        .estimate_control_rgb8(img.as_raw(), w, h)
        .map_err(|error| WorkerError::Engine(format!("depth estimate: {error}")))?;
    image::RgbImage::from_raw(w, h, control).ok_or_else(|| {
        WorkerError::Engine("depth estimator returned a mis-sized control buffer".to_owned())
    })
}
