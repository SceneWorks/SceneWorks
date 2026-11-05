//! Shared raster decode and platform backstop (sc-6143, F-172).
//!
//! The workspace native codec set handles PNG/JPEG/WebP/BMP/GIF/TIFF. AVIF and HEIC/HEIF
//! use the bounded platform transcoder, as does the explicit OpenEXR tone-map backstop. New
//! uploads normalize at import; older assets and paths that skip normalization decode here.
//!
//! [`decode_image_any`] is a drop-in replacement for `image::open`: same signature, same error type.
//! It first tries the fast native decode (and a content-sniffed in-memory decode, which also handles
//! a supported image whose extension is wrong or absent), then, only for a recognized
//! unsupported-but-valid format, transcodes to a temp PNG via the shared
//! [`crate::media_convert`] routine and decodes that. So a job degrades to
//! "transcode + run" instead of failing, and the format whack-a-mole lives in one place.

use std::io;
use std::path::Path;

use crate::media_convert::{sniff_image_kind, transcode_to_png};
use image::{DynamicImage, ImageError, ImageResult};

/// Decode an image file, transcoding a valid-but-unsupported format to PNG on the fly. Drop-in for
/// `image::open`.
pub fn decode_image_any(path: impl AsRef<Path>) -> ImageResult<DynamicImage> {
    decode_file(path.as_ref(), |path| image::open(path), true, false)
}

/// Decode with a caller's native pixel policy (for example EXIF orientation). The callback is
/// applied to both the source and the transcoded PNG; transcode failures retain their diagnostic.
pub fn decode_image_with<F>(path: &Path, decode: F) -> ImageResult<DynamicImage>
where
    F: Fn(&Path) -> ImageResult<DynamicImage>,
{
    decode_file(path, decode, false, true)
}

fn decode_file<F>(
    path: &Path,
    decode: F,
    retry_memory: bool,
    expose_transcode_error: bool,
) -> ImageResult<DynamicImage>
where
    F: Fn(&Path) -> ImageResult<DynamicImage>,
{
    let open_error = match decode(path) {
        Ok(image) => return Ok(image),
        Err(error) => error,
    };
    let kind = if retry_memory {
        let bytes = std::fs::read(path).map_err(ImageError::IoError)?;
        if let Ok(image) = image::load_from_memory(&bytes) {
            return Ok(image);
        }
        sniff_image_kind(&bytes)
    } else {
        crate::media_convert::sniff_image_kind_at(path)
    };
    let Some(kind) = kind else {
        return Err(open_error);
    };
    if kind.is_natively_supported() {
        return Err(open_error);
    }
    let result = run_transcode_decode_with(|out| transcode_to_png(path, out), decode);
    if expose_transcode_error {
        result
    } else {
        result.or(Err(open_error))
    }
}

/// Decode image bytes, transcoding a valid-but-unsupported format to PNG on the fly. Drop-in for
/// `image::load_from_memory` (used where the worker already holds the bytes — e.g. a no-extension
/// temp upload sniffed from memory).
pub fn decode_image_bytes_any(bytes: &[u8]) -> ImageResult<DynamicImage> {
    match image::load_from_memory(bytes) {
        Ok(image) => Ok(image),
        Err(memory_error) => match transcode_then_decode_bytes(bytes) {
            Some(Ok(image)) => Ok(image),
            Some(Err(_)) | None => Err(memory_error),
        },
    }
}

fn transcode_then_decode_bytes(bytes: &[u8]) -> Option<ImageResult<DynamicImage>> {
    let kind = sniff_image_kind(bytes)?;
    if kind.is_natively_supported() {
        return None;
    }
    Some(run_transcode_decode_with(
        |out| {
            let dir = out.parent().unwrap_or_else(|| Path::new("."));
            let src = dir.join("source.bin");
            std::fs::write(&src, bytes)
                .map_err(|error| crate::media_convert::TranscodeError(error.to_string()))?;
            transcode_to_png(&src, out)
        },
        |path| image::open(path),
    ))
}

/// Run a transcode (writing a PNG into a temp dir) and decode the result, logging and downgrading a
/// transcode failure so the caller can fall back to the original decoder error.
fn run_transcode_decode_with<F, D>(transcode: F, decode: D) -> ImageResult<DynamicImage>
where
    F: FnOnce(&Path) -> Result<(), crate::media_convert::TranscodeError>,
    D: FnOnce(&Path) -> ImageResult<DynamicImage>,
{
    let dir = tempfile::tempdir().map_err(ImageError::IoError)?;
    let out = dir.path().join("decoded.png");
    if let Err(error) = transcode(&out) {
        tracing::error!(
            event = "image_decode_transcode_failed",
            error = %error,
            "failed to transcode a decoded image"
        );
        return Err(ImageError::IoError(io::Error::other(error.to_string())));
    }
    decode(&out)
}

/// Explicit resource and format policy for untrusted dataset/catalog bytes. The native decoder
/// keeps its source pixels/orientation; network and prepared-bundle policy remains caller-owned.
#[derive(Debug, Clone, Copy)]
pub struct ImageDecodeLimits {
    pub max_bytes: usize,
    pub max_width: u32,
    pub max_height: u32,
    pub max_pixels: u64,
    pub max_alloc: u64,
    pub catalog_formats_only: bool,
}

/// Check encoded size, format and header dimensions before allocating pixels, then apply the same
/// axis/allocation limits to the actual decoder. Catalog callers deliberately forbid backstops.
pub fn decode_image_bounded(
    bytes: &[u8],
    limits: ImageDecodeLimits,
) -> ImageResult<(DynamicImage, image::ImageFormat)> {
    let invalid =
        |message| ImageError::IoError(io::Error::new(io::ErrorKind::InvalidData, message));
    if bytes.is_empty() || bytes.len() > limits.max_bytes {
        return Err(invalid("image exceeds encoded byte limit"));
    }
    let mut reader = image::ImageReader::new(io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(ImageError::IoError)?;
    let format = reader
        .format()
        .ok_or_else(|| invalid("image format is unknown"))?;
    if limits.catalog_formats_only
        && !matches!(
            format,
            image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP
        )
    {
        return Err(invalid("image format must be PNG, JPEG, or WebP"));
    }
    let mut decoder_limits = image::Limits::default();
    decoder_limits.max_image_width = Some(limits.max_width);
    decoder_limits.max_image_height = Some(limits.max_height);
    decoder_limits.max_alloc = Some(limits.max_alloc);
    let mut header_limits = decoder_limits.clone();
    header_limits.max_image_width = None;
    header_limits.max_image_height = None;
    reader.limits(header_limits);
    let (width, height) = reader.into_dimensions()?;
    if width == 0
        || height == 0
        || width > limits.max_width
        || height > limits.max_height
        || u64::from(width) * u64::from(height) > limits.max_pixels
    {
        return Err(invalid("image dimensions exceed configured limits"));
    }
    let mut reader = image::ImageReader::with_format(io::Cursor::new(bytes), format);
    reader.limits(decoder_limits);
    let image = reader.decode()?;
    // Defend against a decoder/header disagreement without changing the caller's format policy.
    if image.width() != width || image.height() != height {
        return Err(invalid("image decoded dimensions disagree with header"));
    }
    Ok((image, format))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ImageDecodeLimits {
        ImageDecodeLimits {
            max_bytes: 1024 * 1024,
            max_width: 64,
            max_height: 64,
            max_pixels: 4096,
            max_alloc: 1024 * 1024,
            catalog_formats_only: false,
        }
    }

    #[test]
    fn bounded_native_formats_and_catalog_subset_are_preserved() {
        for format in [
            image::ImageFormat::Png,
            image::ImageFormat::Jpeg,
            image::ImageFormat::WebP,
            image::ImageFormat::Bmp,
            image::ImageFormat::Gif,
            image::ImageFormat::Tiff,
        ] {
            let mut encoded = io::Cursor::new(Vec::new());
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(3, 2, image::Rgb([10, 20, 30])))
                .write_to(&mut encoded, format)
                .expect("native fixture encodes");
            let bytes = encoded.into_inner();
            let (decoded, detected) =
                decode_image_bounded(&bytes, policy()).expect("native decode");
            assert_eq!(detected, format);
            assert_eq!((decoded.width(), decoded.height()), (3, 2));
            let catalog = ImageDecodeLimits {
                catalog_formats_only: true,
                ..policy()
            };
            assert_eq!(
                decode_image_bounded(&bytes, catalog).is_ok(),
                matches!(
                    format,
                    image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP
                )
            );
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("wrong.extension");
            std::fs::write(&path, &bytes).unwrap();
            assert_eq!(decode_image_any(&path).unwrap().width(), 3);
            assert_eq!(decode_image_bytes_any(&bytes).unwrap().height(), 2);
        }
    }

    #[test]
    fn bounded_decode_preserves_alpha_and_refuses_each_resource_boundary() {
        let pixels = image::RgbaImage::from_pixel(3, 2, image::Rgba([10, 20, 30, 40]));
        let mut encoded = io::Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(pixels.clone())
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let bytes = encoded.into_inner();
        assert_eq!(
            decode_image_bounded(&bytes, policy()).unwrap().0.to_rgba8(),
            pixels
        );
        for limits in [
            ImageDecodeLimits {
                max_bytes: bytes.len() - 1,
                ..policy()
            },
            ImageDecodeLimits {
                max_width: 2,
                ..policy()
            },
            ImageDecodeLimits {
                max_height: 1,
                ..policy()
            },
            ImageDecodeLimits {
                max_pixels: 5,
                ..policy()
            },
            ImageDecodeLimits {
                max_alloc: 1,
                ..policy()
            },
        ] {
            assert!(decode_image_bounded(&bytes, limits).is_err(), "{limits:?}");
        }
        assert!(decode_image_bounded(&bytes[..bytes.len() / 2], policy()).is_err());
        assert!(decode_image_bounded(b"", policy()).is_err());
        assert!(decode_image_bounded(b"not an image", policy()).is_err());
    }

    #[test]
    fn failed_transcode_preserves_worker_error_and_quality_diagnostic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.heic");
        std::fs::write(&path, b"\0\0\0\x18ftypheic\0\0\0\0heic").unwrap();
        let original = image::open(&path).unwrap_err().to_string();
        assert_eq!(decode_image_any(&path).unwrap_err().to_string(), original);
        assert!(matches!(
            decode_image_with(&path, |path| image::open(path)).unwrap_err(),
            ImageError::IoError(_)
        ));
    }

    /// A supported image whose path has no extension still decodes (the `image::open`-fails →
    /// in-memory-decode fallback), without any transcode.
    #[test]
    fn decodes_supported_image_with_no_extension() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("no_extension_here");
        let png = image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]));
        png.save_with_format(&path, image::ImageFormat::Png)
            .expect("write png without a .png extension");

        let decoded = decode_image_any(&path).expect("decodes by content");
        assert_eq!(decoded.to_rgb8().dimensions(), (2, 2));
        assert_eq!(
            decode_image_bytes_any(&std::fs::read(&path).unwrap())
                .expect("bytes decode")
                .to_rgb8()
                .dimensions(),
            (2, 2)
        );
    }

    /// A genuinely unreadable/garbage file surfaces an error (not a panic, not a transcode hang).
    #[test]
    fn unrecognized_bytes_error_out() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("junk.png");
        std::fs::write(&path, b"this is not an image").expect("write junk");
        assert!(decode_image_any(&path).is_err());
        assert!(decode_image_bytes_any(b"this is not an image").is_err());
    }

    /// End-to-end backstop: a valid-but-unsupported source decodes via the transcode path.
    /// macOS-only because it relies on `sips`; the ffmpeg path is identical.
    ///
    /// The fixture must be a format that [`sniff_image_kind`] recognizes (the transcode is gated on
    /// a known kind) AND that the compiled `image` build cannot decode. This used to hand-build a
    /// BMP, which passed under `cargo test -p sceneworks-worker` but FAILED under a workspace
    /// `cargo test`: cargo unifies features for the shared `image` 0.25 across workspace members and
    /// `apps/rust-api` enables `bmp`, so every real build decodes BMP natively and the precondition
    /// stopped holding. HEIC is immune — `image` 0.25 ships no HEIC decoder at any feature level —
    /// so this no longer depends on which crate enables what.
    #[cfg(target_os = "macos")]
    #[test]
    fn decodes_unsupported_heic_via_transcode() {
        let dir = tempfile::tempdir().expect("temp dir");
        let seed = dir.path().join("seed.png");
        image::RgbImage::from_raw(1, 1, vec![0x20, 0x40, 0x80])
            .expect("1x1 rgb buffer")
            .save_with_format(&seed, image::ImageFormat::Png)
            .expect("seed png writes");
        let path = dir.path().join("pixel.heic");
        let converted = std::process::Command::new("sips")
            .args(["-s", "format", "heic"])
            .arg(&seed)
            .arg("--out")
            .arg(&path)
            .output()
            .expect("sips runs");
        assert!(
            converted.status.success() && path.exists(),
            "sips must produce the HEIC fixture: {}",
            String::from_utf8_lossy(&converted.stderr)
        );

        // The transcode is only reachable for a sniffed kind.
        assert_eq!(
            sniff_image_kind(&std::fs::read(&path).expect("read fixture")),
            Some(crate::media_convert::ImageKind::Heif)
        );
        // The worker's image build can't decode HEIC directly...
        assert!(image::open(&path).is_err());
        // ...but the backstop transcodes it.
        let decoded = decode_image_any(&path).expect("HEIC decodes via transcode");
        assert_eq!(decoded.to_rgb8().dimensions(), (1, 1));
        assert_eq!(
            decode_image_bytes_any(&std::fs::read(&path).unwrap())
                .unwrap()
                .width(),
            1
        );
        assert_eq!(
            decode_image_with(&path, |path| image::open(path))
                .unwrap()
                .width(),
            1
        );
    }
}
