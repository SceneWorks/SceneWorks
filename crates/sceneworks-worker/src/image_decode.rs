//! Image-decode backstop (sc-6143).
//!
//! The worker declares `image` as `png`/`jpeg`/`webp`, so a valid AVIF/HEIC/HEIF (and, in a build
//! whose workspace-wide `image` feature union lacks them, TIFF/BMP/GIF) asset fails to decode
//! (`The image format Avif is not supported`). Note the decodable set is the union across all
//! workspace members, not this crate's own declaration — `apps/rust-api` enabling `bmp` makes BMP
//! decodable here too. Import-time
//! normalization ([`sceneworks_core::project_store`]) converts new uploads to PNG, but assets that
//! predate that change — or arrive by a path that skips import normalization — would still fail at
//! the ~dozen worker decode sites.
//!
//! [`decode_image_any`] is a drop-in replacement for `image::open`: same signature, same error type.
//! It first tries the fast native decode (and a content-sniffed in-memory decode, which also handles
//! a supported image whose extension is wrong or absent), then, only for a recognized
//! unsupported-but-valid format, transcodes to a temp PNG via the shared
//! [`sceneworks_core::media_convert`] routine and decodes that. So a job degrades to
//! "transcode + run" instead of failing, and the format whack-a-mole lives in one place.

use std::io;
use std::path::Path;

use image::{DynamicImage, ImageError, ImageResult};
use sceneworks_core::media_convert::{sniff_image_kind, transcode_to_png};

/// Decode an image file, transcoding a valid-but-unsupported format to PNG on the fly. Drop-in for
/// `image::open`.
// Used on every always-compiled decode path (video/caption); the conditional allow only silences
// the bare (non-macOS, non-candle) lib build where the remaining callers are cfg'd out.
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn decode_image_any(path: impl AsRef<Path>) -> ImageResult<DynamicImage> {
    let path = path.as_ref();
    match image::open(path) {
        Ok(image) => Ok(image),
        Err(open_error) => {
            // Read once; reuse for the in-memory decode and the format sniff.
            let bytes = std::fs::read(path).map_err(ImageError::IoError)?;
            // A supported image with a wrong/absent extension (the no-extension temp-upload case)
            // decodes here without shelling out — `image::open` keys off the extension.
            if let Ok(image) = image::load_from_memory(&bytes) {
                return Ok(image);
            }
            match transcode_then_decode(path, &bytes) {
                Some(Ok(image)) => Ok(image),
                // Transcode attempted but failed: surface the original decoder error (keeps the
                // familiar message) — the warning was already logged.
                Some(Err(_)) | None => Err(open_error),
            }
        }
    }
}

/// Decode image bytes, transcoding a valid-but-unsupported format to PNG on the fly. Drop-in for
/// `image::load_from_memory` (used where the worker already holds the bytes — e.g. a no-extension
/// temp upload sniffed from memory).
#[cfg_attr(
    all(not(target_os = "macos"), not(feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn decode_image_bytes_any(bytes: &[u8]) -> ImageResult<DynamicImage> {
    match image::load_from_memory(bytes) {
        Ok(image) => Ok(image),
        Err(memory_error) => match transcode_then_decode_bytes(bytes) {
            Some(Ok(image)) => Ok(image),
            Some(Err(_)) | None => Err(memory_error),
        },
    }
}

/// `Some` when `bytes` are a recognized unsupported-but-valid format we attempted to transcode (the
/// inner `Result` is the decode outcome); `None` when the format is not one we transcode.
fn transcode_then_decode(src: &Path, bytes: &[u8]) -> Option<ImageResult<DynamicImage>> {
    let kind = sniff_image_kind(bytes)?;
    if kind.is_natively_supported() {
        return None; // would have decoded above already
    }
    Some(run_transcode_decode(|out| transcode_to_png(src, out)))
}

fn transcode_then_decode_bytes(bytes: &[u8]) -> Option<ImageResult<DynamicImage>> {
    let kind = sniff_image_kind(bytes)?;
    if kind.is_natively_supported() {
        return None;
    }
    Some(run_transcode_decode(|out| {
        let dir = out.parent().unwrap_or_else(|| Path::new("."));
        let src = dir.join("source.bin");
        std::fs::write(&src, bytes)
            .map_err(|error| sceneworks_core::media_convert::TranscodeError(error.to_string()))?;
        transcode_to_png(&src, out)
    }))
}

/// Run a transcode (writing a PNG into a temp dir) and decode the result, logging and downgrading a
/// transcode failure so the caller can fall back to the original decoder error.
fn run_transcode_decode<F>(transcode: F) -> ImageResult<DynamicImage>
where
    F: FnOnce(&Path) -> Result<(), sceneworks_core::media_convert::TranscodeError>,
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
    image::open(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            Some(sceneworks_core::media_convert::ImageKind::Heif)
        );
        // The worker's image build can't decode HEIC directly...
        assert!(image::open(&path).is_err());
        // ...but the backstop transcodes it.
        let decoded = decode_image_any(&path).expect("HEIC decodes via transcode");
        assert_eq!(decoded.to_rgb8().dimensions(), (1, 1));
    }
}
