use super::Image;
use crate::WorkerError;
use crate::WorkerResult;
// Image fit/crop/pad geometry shared by the MLX edit handlers (flux2/qwen/sdxl/kolors/sensenova/
// zimage), the candle edit handlers (*_edit_candle.rs), and the video I2V resolve paths
// (video_jobs.rs). Kept in base.rs — included on macOS AND the `backend-candle` lane (and nowhere
// else) — so `crate::image_jobs::fit_engine_image` resolves on exactly the lanes that call it. Moved
// here from the macOS MLX flux2.rs (sc-6231; the sc-6139 fit-mode refactor left the shared helper macOS-gated, which
// broke the candle build because video_jobs.rs / the candle edit handlers call it). No `#[cfg]` here:
// availability follows base.rs's own include cfg, which matches the callers'.

/// Resize an RGB image to exactly `width`×`height` honoring `mode` without distorting it
/// (parity with Python `fit_image`, RGB path only — no inpaint mask exists on the MLX
/// FLUX.2 edit path, so `outpaint` degrades to `pad` geometry):
///   - `crop`:    scale to COVER (short edge fits), center-crop the overflow.
///   - `pad`/`outpaint`: scale to CONTAIN (long edge fits), center on a black canvas.
///   - `stretch`: legacy non-aspect-preserving resize.
///
/// The pad/outpaint arm's contain geometry is the engine's [`gen_core::imageops::contain_box`]
/// (sc-8824) — the SINGLE source of truth shared with the outpaint mask
/// ([`gen_core::imageops::outpaint_border_mask`] calls the same `contain_box`), so the letterboxed
/// kept-rect and the mask's keep-rect are pixel-identical. It rounds half-to-even in f64 (matching
/// Python `round`) and returns i32 offsets; the old local copy rounded half-away-from-zero in f32,
/// which could disagree by a pixel at an exact `.5` and desync fit vs. mask on outpaint edges.
pub(super) fn fit_rgb(
    source: &image::RgbImage,
    width: u32,
    height: u32,
    mode: &str,
) -> image::RgbImage {
    use image::imageops::FilterType::Lanczos3;
    let width = width.max(1);
    let height = height.max(1);
    let (src_w, src_h) = (source.width(), source.height());
    match mode {
        "stretch" => image::imageops::resize(source, width, height, Lanczos3),
        "crop" => {
            let ratio = (width as f32 / src_w as f32).max(height as f32 / src_h as f32);
            // Ceil so the scaled image always fully covers the target before cropping.
            let new_w = width.max((src_w as f32 * ratio).ceil() as u32);
            let new_h = height.max((src_h as f32 * ratio).ceil() as u32);
            let resized = image::imageops::resize(source, new_w, new_h, Lanczos3);
            let left = (new_w - width) / 2;
            let top = (new_h - height) / 2;
            image::imageops::crop_imm(&resized, left, top, width, height).to_image()
        }
        // "pad" / "outpaint": contain + center on a black canvas (letterbox). The engine's
        // `contain_box` is the shared geometry the outpaint mask also uses, so fit + mask agree.
        _ => {
            let (new_w, new_h, left, top) =
                gen_core::imageops::contain_box(src_w, src_h, width, height);
            let resized = image::imageops::resize(source, new_w.max(1), new_h.max(1), Lanczos3);
            let mut canvas = image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0]));
            image::imageops::overlay(&mut canvas, &resized, left as i64, top as i64);
            canvas
        }
    }
}

/// Fit an engine [`Image`] (RGB8) to `width`×`height` by `mode` via [`fit_rgb`].
/// `pub(crate)` so the video I2V resolve paths (`video_jobs.rs`, sc-6139) can pre-fit a
/// starting image to the output dims with the same crop/pad geometry as the image-edit lane.
pub(crate) fn fit_engine_image(
    source: Image,
    width: u32,
    height: u32,
    mode: &str,
) -> WorkerResult<Image> {
    let rgb =
        image::RgbImage::from_raw(source.width, source.height, source.pixels).ok_or_else(|| {
            WorkerError::InvalidPayload("edit source buffer size mismatch".to_owned())
        })?;
    let fitted = fit_rgb(&rgb, width, height, mode);
    Ok(Image {
        width: fitted.width(),
        height: fitted.height(),
        pixels: fitted.into_raw(),
    })
}
