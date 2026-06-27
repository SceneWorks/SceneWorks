//! Canny edge-map preprocessor for the Fun-Controlnet-Union canny head
//! (epic 8236, sc-8240). A native, CPU, cross-platform model-agnostic
//! preprocessor — sibling of `openpose_skeleton`: arbitrary input image →
//! canny control image for `ControlKind::Canny`.
//!
//! Like the pose skeleton, this is pure raster (no GPU / MLX), so it builds and
//! runs everywhere and is testable on any platform; only the MLX control
//! generation that *consumes* the edge map is macOS-gated.
//!
//! Output contract: an [`image::RgbImage`], byte-for-byte the same type the pose
//! preprocessor's `draw_wholebody` returns and the `ControlKind::Canny` path
//! consumes (saved as a PNG control image / fed to the control branch). The edge
//! map is single-channel (white edges on black) broadcast across all three RGB
//! channels, matching the standard ControlNet canny convention (edges drawn
//! `255,255,255` on a `0,0,0` field).

use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::edges::canny;

/// Default low hysteresis threshold (weak-edge gate). 50.0 / 100.0 is the de-facto
/// ControlNet canny default (the `controlnet_aux` / diffusers `CannyDetector`
/// pairing), tuned for 8-bit gradient magnitudes.
pub const DEFAULT_LOW_THRESHOLD: f32 = 50.0;

/// Default high hysteresis threshold (strong-edge gate). See [`DEFAULT_LOW_THRESHOLD`].
pub const DEFAULT_HIGH_THRESHOLD: f32 = 100.0;

/// Convert an input image to a luma (grayscale) buffer for edge detection.
///
/// Standard Rec. 601 luma weighting (`image`'s `to_luma8`), matching what
/// `cv2.cvtColor(..., COLOR_RGB2GRAY)` produces upstream so edge magnitudes land
/// where the canny head was trained.
fn to_gray(img: &RgbImage) -> GrayImage {
    image::DynamicImage::ImageRgb8(img.clone()).to_luma8()
}

/// Broadcast a single-channel edge map (white edges on black) to an RGB control
/// image: each gray value `v` becomes `[v, v, v]`. Keeps the exact `RgbImage`
/// output contract the pose preprocessor and `ControlKind::Canny` path share.
fn gray_to_rgb(edges: &GrayImage) -> RgbImage {
    let (w, h) = (edges.width(), edges.height());
    let mut out = RgbImage::new(w, h);
    for (x, y, &Luma([v])) in edges.enumerate_pixels() {
        out.put_pixel(x, y, Rgb([v, v, v]));
    }
    out
}

/// Run canny edge detection on `img` with explicit hysteresis thresholds and
/// render the result as an RGB control image (white edges on black).
///
/// `low_threshold` / `high_threshold` are the weak/strong gradient-magnitude
/// gates (8-bit scale); a higher pair yields fewer edge pixels. `imageproc`'s
/// `canny` expects `low <= high`; callers should keep that ordering (the standard
/// ControlNet defaults [`DEFAULT_LOW_THRESHOLD`] / [`DEFAULT_HIGH_THRESHOLD`] do).
///
/// Output dimensions match the input exactly, and the type is the same
/// [`RgbImage`] the pose preprocessor emits — drop-in for the `ControlKind::Canny`
/// consumption wired by the control driver (sc-8243 / extend stories).
pub fn canny_control_image(img: &RgbImage, low_threshold: f32, high_threshold: f32) -> RgbImage {
    let gray = to_gray(img);
    let edges = canny(&gray, low_threshold, high_threshold);
    gray_to_rgb(&edges)
}

/// [`canny_control_image`] with the standard ControlNet canny thresholds
/// ([`DEFAULT_LOW_THRESHOLD`] / [`DEFAULT_HIGH_THRESHOLD`]).
pub fn canny_control_image_default(img: &RgbImage) -> RgbImage {
    canny_control_image(img, DEFAULT_LOW_THRESHOLD, DEFAULT_HIGH_THRESHOLD)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A black canvas with a filled white rectangle. The four rectangle borders are
    /// the only sharp gradients, so canny must fire along them and leave the flat
    /// interior + flat exterior empty.
    fn rect_fixture(w: u32, h: u32, rect: (u32, u32, u32, u32)) -> RgbImage {
        let (rx, ry, rw, rh) = rect;
        let mut img = RgbImage::new(w, h);
        for y in ry..ry + rh {
            for x in rx..rx + rw {
                img.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
        img
    }

    /// Count white (edge) pixels in an edge-map control image.
    fn edge_count(img: &RgbImage) -> usize {
        img.pixels().filter(|p| p.0 == [255, 255, 255]).count()
    }

    #[test]
    fn output_is_rgb_same_dimensions_and_binary() {
        let img = rect_fixture(64, 48, (16, 12, 24, 20));
        let out = canny_control_image_default(&img);
        assert_eq!(
            (out.width(), out.height()),
            (64, 48),
            "dims must match input"
        );
        // Edge map is a single-channel value broadcast to RGB: every pixel is r==g==b.
        assert!(
            out.pixels().all(|p| p.0[0] == p.0[1] && p.0[1] == p.0[2]),
            "control image must be grayscale broadcast to RGB"
        );
    }

    #[test]
    fn fires_on_a_sharp_edge_and_leaves_interior_empty() {
        // A centered 40x40 white rectangle in an 80x80 black field.
        let (rx, ry, rw, rh) = (20u32, 20u32, 40u32, 40u32);
        let img = rect_fixture(80, 80, (rx, ry, rw, rh));
        let out = canny_control_image_default(&img);

        // Edges exist.
        let edges = edge_count(&out);
        assert!(
            edges > 0,
            "canny must produce edge pixels on a sharp boundary"
        );

        // Boundary band (within 2px of the rect outline) holds essentially all edges;
        // the flat interior (well inside the rect) and flat exterior (well outside) are
        // empty. Sample the dead-center of the rectangle and a far-corner background pixel.
        let cx = rx + rw / 2;
        let cy = ry + rh / 2;
        assert_eq!(
            out.get_pixel(cx, cy).0,
            [0, 0, 0],
            "flat rectangle interior must have no edges"
        );
        assert_eq!(
            out.get_pixel(2, 2).0,
            [0, 0, 0],
            "flat background must have no edges"
        );

        // Every edge pixel must lie within a 2px band around the rectangle outline.
        let near_border = |x: u32, y: u32| -> bool {
            let x = x as i64;
            let y = y as i64;
            let (l, r) = (rx as i64, (rx + rw - 1) as i64);
            let (t, b) = (ry as i64, (ry + rh - 1) as i64);
            let near_v = (x - l).abs() <= 2 || (x - r).abs() <= 2;
            let near_h = (y - t).abs() <= 2 || (y - b).abs() <= 2;
            let in_x = (l - 2..=r + 2).contains(&x);
            let in_y = (t - 2..=b + 2).contains(&y);
            (near_v && in_y) || (near_h && in_x)
        };
        for (x, y, p) in out.enumerate_pixels() {
            if p.0 == [255, 255, 255] {
                assert!(
                    near_border(x, y),
                    "edge pixel ({x},{y}) is not on the rectangle boundary"
                );
            }
        }
    }

    #[test]
    fn higher_threshold_yields_fewer_or_equal_edges() {
        // A gradient ramp + a sharp step gives a mix of weak and strong edges so the
        // threshold actually gates something. Left half ramps 0->255, right half is a
        // hard white block (a strong edge at the seam, weak edges across the ramp).
        let (w, h) = (96u32, 64u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 {
                    ((x as f32 / (w as f32 / 2.0)) * 255.0) as u8
                } else {
                    255
                };
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }

        let low = canny_control_image(&img, 20.0, 40.0);
        let high = canny_control_image(&img, 120.0, 240.0);
        let low_edges = edge_count(&low);
        let high_edges = edge_count(&high);

        assert!(
            low_edges > 0,
            "low thresholds must detect edges on the ramp"
        );
        assert!(
            high_edges <= low_edges,
            "higher thresholds must not produce more edges (low={low_edges}, high={high_edges})"
        );
        assert!(
            high_edges < low_edges,
            "higher thresholds should drop weak ramp edges (low={low_edges}, high={high_edges})"
        );
    }

    #[test]
    fn flat_image_has_no_edges() {
        let img = RgbImage::from_pixel(32, 32, Rgb([128, 128, 128]));
        let out = canny_control_image_default(&img);
        assert_eq!(edge_count(&out), 0, "a flat image has no edges");
    }
}
