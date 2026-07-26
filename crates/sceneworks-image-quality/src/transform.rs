//! One-tap image fixes that rewrite an image's pixels (sc-6539): smart-crop and EXIF-strip. Both
//! decode through [`load_oriented`] so the EXIF orientation tag is baked into the pixels first, then
//! re-encode as PNG — which carries no metadata, so the result is upright + clean. The orchestration
//! (resolve items → transform → re-point) lives in the API; these are the pure pixel ops.

use std::path::Path;

use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use sceneworks_core::dataset_quality::plan_smart_crop;

/// Decode an image and bake in its EXIF orientation. `image` 0.25 does NOT auto-apply orientation on
/// decode, and supported uploads are stored byte-for-byte (a phone JPEG keeps its orientation tag),
/// so every decode in the pipeline currently ignores it. Reading the tag here and applying it makes
/// the *pixels* upright — so a subsequent re-encode that drops the tag (PNG carries none) corrects the
/// image rather than silently rotating it. A format with no orientation (PNG) yields a no-op.
///
/// A valid-but-unsupported source (AVIF/HEIC/HEIF, and TIFF/BMP/GIF in a build whose `image` feature
/// union lacks them) is transcoded to a temp PNG first via
/// [`crate::decode_transcoding`] — the transcoder (ImageIO `sips` / `ffmpeg`) bakes orientation, and
/// PNG carries no tag, so the result is already upright (the subsequent `apply_orientation` is a
/// no-op). This is what keeps the Dataset Doctor's smart-crop / EXIF-strip fixes from failing on an
/// AVIF asset that skipped import normalization.
pub fn load_oriented(path: &Path) -> image::ImageResult<DynamicImage> {
    crate::decode_transcoding(path, load_oriented_native)
}

/// The native oriented decode — the fast path, and the closure re-run on the transcoded PNG.
fn load_oriented_native(path: &Path) -> image::ImageResult<DynamicImage> {
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image)
}

/// EXIF-strip fix: write an upright, metadata-free PNG copy of `src` to `out` (drops EXIF/GPS/ICC).
pub fn write_metadata_stripped(src: &Path, out: &Path) -> image::ImageResult<()> {
    load_oriented(src)?.save_with_format(out, ImageFormat::Png)
}

/// Smart-crop fix: trim `src` toward a trainable aspect (crop-loss ≤ `target_crop_loss`) and write the
/// result as PNG to `out`. The crop is planned on the *oriented* dimensions (so a rotated source crops
/// the correct axis) — [`plan_smart_crop`] returns `None` when no crop is needed, in which case the
/// oriented image is written unchanged (still upright + metadata-stripped). Returns whether a crop was
/// applied. `bias` is an optional normalized focal point (`None` centers — the saliency-free default).
pub fn write_smart_cropped(
    src: &Path,
    target_crop_loss: f64,
    bias: Option<(f64, f64)>,
    out: &Path,
) -> image::ImageResult<bool> {
    let image = load_oriented(src)?;
    match plan_smart_crop(image.width(), image.height(), target_crop_loss, bias) {
        Some(rect) => {
            image
                .crop_imm(rect.x, rect.y, rect.width, rect.height)
                .save_with_format(out, ImageFormat::Png)?;
            Ok(true)
        }
        None => {
            image.save_with_format(out, ImageFormat::Png)?;
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageReader, RgbImage};

    fn write_png(dir: &std::path::Path, name: &str, w: u32, h: u32) -> std::path::PathBuf {
        let mut img = RgbImage::new(w, h);
        for (x, y, px) in img.enumerate_pixels_mut() {
            *px = image::Rgb([(x % 256) as u8, (y % 256) as u8, 128]);
        }
        let path = dir.join(name);
        img.save_with_format(&path, ImageFormat::Png).unwrap();
        path
    }

    fn dims(path: &std::path::Path) -> (u32, u32) {
        ImageReader::open(path)
            .unwrap()
            .with_guessed_format()
            .unwrap()
            .into_dimensions()
            .unwrap()
    }

    #[test]
    fn smart_crop_trims_an_extreme_aspect_to_png() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_png(dir.path(), "wide.png", 200, 64);
        let out = dir.path().join("wide-cropped.png");
        let cropped = write_smart_cropped(&src, 0.25, None, &out).unwrap();
        assert!(cropped, "200x64 (crop-loss 0.68) needs a crop");
        let (w, h) = dims(&out);
        assert_eq!(h, 64, "short edge kept");
        assert!((64..200).contains(&w));
        assert!(
            f64::from(w - h) / f64::from(w) < 0.35,
            "clears the crop-loss flag"
        );
    }

    #[test]
    fn smart_crop_leaves_a_square_uncropped_but_re_encodes() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_png(dir.path(), "square.png", 100, 100);
        let out = dir.path().join("square-out.png");
        let cropped = write_smart_cropped(&src, 0.25, None, &out).unwrap();
        assert!(!cropped, "square needs no crop");
        assert_eq!(
            dims(&out),
            (100, 100),
            "written unchanged (oriented + stripped)"
        );
    }

    #[test]
    fn strip_metadata_writes_a_valid_same_size_png() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_png(dir.path(), "meta.png", 48, 32);
        let out = dir.path().join("meta-stripped.png");
        write_metadata_stripped(&src, &out).unwrap();
        assert_eq!(dims(&out), (48, 32));
        // re-decodes cleanly (a valid PNG).
        assert!(load_oriented(&out).is_ok());
    }

    #[test]
    fn load_oriented_is_a_no_op_for_an_unoriented_png() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_png(dir.path(), "plain.png", 70, 50);
        let img = load_oriented(&src).unwrap();
        assert_eq!((img.width(), img.height()), (70, 50));
    }

    /// The DataDoctor regression: a valid-but-unsupported source no longer errors in the transform
    /// path — it transcodes via `sips` and the smart-crop / EXIF-strip fixes run. macOS-only because
    /// it relies on the always-present `sips`; the ffmpeg path is identical.
    ///
    /// The fixture must satisfy BOTH gates or it proves nothing:
    ///  1. `sniff_image_kind_at` has to recognize it — [`crate::decode_transcoding`] only shells out
    ///     for a sniffed [`ImageKind`], so an unsniffable format (TGA, QOI, …) would just surface the
    ///     native decode error and never reach the transcoder.
    ///  2. The compiled `image` build genuinely cannot decode it.
    ///
    /// This used to hand-build a BMP, which made the test pass under
    /// `cargo test -p sceneworks-image-quality` but FAIL under a workspace `cargo test` (sc-15028):
    /// Cargo unified features for the shared `image` 0.25 across members, another member enabled
    /// `bmp`, and so every shipped build decoded BMP natively and gate 2 stopped holding.
    /// sc-15052 removed that particular trapdoor — `image` is declared once in the root
    /// `[workspace.dependencies]` (bmp/gif/jpeg/png/tiff/webp), so this crate's declared features
    /// and its compiled features are now the same thing in either scope. Gate 2 must still not
    /// lean on a format merely being absent from that list: the list is workspace-wide and one
    /// line away from growing.
    ///
    /// HEIC is immune to that drift: `image` 0.25 ships no HEIC decoder under any feature, so gate 2
    /// cannot be silently invalidated by another crate's Cargo.toml. Generated at test time with the
    /// `sips` this test already depends on, so there is no binary fixture to check in.
    #[cfg(target_os = "macos")]
    #[test]
    fn transform_path_transcodes_an_unsupported_source() {
        let dir = tempfile::tempdir().unwrap();
        let seed = write_png(dir.path(), "seed.png", 1, 1);
        let src = dir.path().join("pixel.heic");
        let converted = std::process::Command::new("sips")
            .args(["-s", "format", "heic"])
            .arg(&seed)
            .arg("--out")
            .arg(&src)
            .output()
            .expect("sips runs");
        assert!(
            converted.status.success() && src.exists(),
            "sips must produce the HEIC fixture: {}",
            String::from_utf8_lossy(&converted.stderr)
        );

        // Gate 1: the transcode fallback is reachable (the bytes sniff as a known kind).
        assert_eq!(
            sceneworks_core::media_convert::sniff_image_kind_at(&src),
            Some(sceneworks_core::media_convert::ImageKind::Heif),
            "the fixture must sniff as a recognized kind or the transcoder is never consulted"
        );
        // Gate 2: the crate's image build genuinely can't decode this directly.
        assert!(load_oriented_native(&src).is_err());

        // But the public path transcodes it, so both one-tap fixes succeed.
        assert!(
            load_oriented(&src).is_ok(),
            "transcodes through load_oriented"
        );

        let stripped = dir.path().join("stripped.png");
        write_metadata_stripped(&src, &stripped).unwrap();
        assert_eq!(dims(&stripped), (1, 1));

        let cropped = dir.path().join("cropped.png");
        write_smart_cropped(&src, 0.25, None, &cropped).unwrap();
        assert_eq!(
            dims(&cropped),
            (1, 1),
            "1x1 is square — no crop, just re-encode"
        );
    }

    #[test]
    fn baking_a_90_degree_orientation_swaps_dims() {
        // The behavior load_oriented relies on for rotated sources: a 90°/270° tag swaps width and
        // height, which is exactly why the crop must be planned on the *oriented* dimensions.
        let mut img = image::DynamicImage::ImageRgb8(RgbImage::new(10, 20));
        img.apply_orientation(image::metadata::Orientation::Rotate90);
        assert_eq!((img.width(), img.height()), (20, 10));
    }
}
