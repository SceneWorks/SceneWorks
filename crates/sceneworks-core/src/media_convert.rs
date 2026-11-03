//! Transcode valid-but-unsupported image formats (AVIF, HEIC/HEIF, TIFF, BMP, GIF) to lossless
//! PNG (sc-6143).
//!
//! The worker's `image` crate is compiled `png`/`jpeg`/`webp`-only, and no pure-Rust HEIC decoder
//! exists at all, so any other format fails to decode anywhere downstream
//! (`The image format Avif is not supported`). Rather than reject a perfectly valid upload we
//! convert it once, losslessly, to PNG — the one format every decode site, thumbnail, and preview
//! already handles.
//!
//! This is the single transcoder routine shared by both layers of the fix:
//! 1. import-time normalization in [`crate::project_store::ProjectStore`] (converts new uploads), and
//! 2. the worker's `decode_image_any` backstop (catches assets that predate the change or arrive by
//!    a path that skips import normalization).
//!
//! Conversion shells out to the platform's always-available decoder — macOS `sips` (ImageIO-backed)
//! with an `ffmpeg` fallback (and `ffmpeg`-only off macOS) — so it pulls in no native image-codec
//! build (libdav1d/nasm/meson) and stays correct on the Windows candle lane.

use crate::media_process::{self, ProcessOutput};
use std::fmt;
use std::path::Path;
#[cfg(test)]
use std::process::Command;
use std::time::Duration;
use tokio::io::AsyncReadExt;

/// Raster image formats recognized by magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageKind {
    /// Decodable directly by the worker's feature-restricted `image` build.
    Png,
    Jpeg,
    WebP,
    /// Valid images the worker cannot decode (no codec compiled / no Rust decoder) → transcode.
    Gif,
    Bmp,
    Tiff,
    Avif,
    /// HEIF container family — covers HEIC (iPhone photos) and plain HEIF.
    Heif,
    /// OpenEXR — scene-linear high-dynamic-range float frames (sc-18790). Unlike every other
    /// member here, its PNG rendition is a **lossy proxy**, not a faithful substitute; see
    /// [`ImageKind::png_transcode_is_lossy`].
    OpenExr,
}

impl ImageKind {
    /// True for the png/jpeg/webp set SceneWorks stores assets in; everything else is transcoded to
    /// PNG on import.
    ///
    /// This is a storage policy, not a read of the compiled `image` features. The workspace build
    /// also decodes bmp/gif/tiff (sc-15052), and those are still transcoded on purpose so stored
    /// assets stay in three formats. Do not "fix" this list to match the feature union.
    pub fn is_natively_supported(self) -> bool {
        matches!(self, ImageKind::Png | ImageKind::Jpeg | ImageKind::WebP)
    }

    /// True when converting this format to PNG **destroys information the file exists to carry**,
    /// so the original must be stored rather than replaced (sc-18790).
    ///
    /// This is the distinction that separates OpenEXR from every other transcoded format here.
    /// AVIF/HEIC/TIFF/BMP/GIF are 8-bit-per-channel display-referred images: PNG holds their
    /// pixels exactly, so import can convert once and keep only the PNG. OpenEXR is scene-linear
    /// **float** — a specular highlight at 50.0 and diffuse white at 1.0 both land on 255 in an
    /// 8-bit PNG, which is precisely the latitude a grading or VFX pipeline needs. Converting and
    /// discarding would leave the user with a tone-mapped proxy and no way back.
    ///
    /// So a lossy-to-transcode format is stored as-is and its PNG rendition is generated as a
    /// **preview derivative** on demand (the `?thumbnail=` route), never as the stored asset.
    /// Download therefore always serves the original bytes.
    pub fn png_transcode_is_lossy(self) -> bool {
        matches!(self, ImageKind::OpenExr)
    }

    /// Canonical `(extension, mime)` for this format — the values to record for a stored asset,
    /// keyed off the detected content rather than the upload's (possibly wrong) extension.
    pub fn canonical(self) -> (&'static str, &'static str) {
        match self {
            ImageKind::Png => ("png", "image/png"),
            ImageKind::Jpeg => ("jpg", "image/jpeg"),
            ImageKind::WebP => ("webp", "image/webp"),
            ImageKind::Gif => ("gif", "image/gif"),
            ImageKind::Bmp => ("bmp", "image/bmp"),
            ImageKind::Tiff => ("tiff", "image/tiff"),
            ImageKind::Avif => ("avif", "image/avif"),
            ImageKind::Heif => ("heic", "image/heic"),
            // `image/x-exr` is the conventional spelling and, critically, an inert
            // `image/` type — the property SAFE_UPLOAD_EXTENSIONS relies on.
            ImageKind::OpenExr => ("exr", "image/x-exr"),
        }
    }

    /// Human-readable label for error/log messages.
    pub fn label(self) -> &'static str {
        match self {
            ImageKind::Png => "PNG",
            ImageKind::Jpeg => "JPEG",
            ImageKind::WebP => "WebP",
            ImageKind::Gif => "GIF",
            ImageKind::Bmp => "BMP",
            ImageKind::Tiff => "TIFF",
            ImageKind::Avif => "AVIF",
            ImageKind::Heif => "HEIC/HEIF",
            ImageKind::OpenExr => "OpenEXR",
        }
    }
}

/// Classify an image by its leading bytes. Content-based, never the file extension — a `.png` that
/// is really AVIF (or a `.jpg` that is really HEIC) is classified by what it actually is. Returns
/// `None` for bytes we don't recognize as one of the handled raster formats (e.g. SVG, or an
/// ISOBMFF stream whose brand is a video, not an image).
pub fn sniff_image_kind(header: &[u8]) -> Option<ImageKind> {
    if header.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(ImageKind::Jpeg);
    }
    if header.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some(ImageKind::Png);
    }
    if header.len() >= 12 && &header[0..4] == b"RIFF" && &header[8..12] == b"WEBP" {
        return Some(ImageKind::WebP);
    }
    if header.starts_with(b"GIF87a") || header.starts_with(b"GIF89a") {
        return Some(ImageKind::Gif);
    }
    // OpenEXR magic number 20000630 (0x01312F76), little-endian on disk.
    if header.starts_with(&[0x76, 0x2F, 0x31, 0x01]) {
        return Some(ImageKind::OpenExr);
    }
    if header.starts_with(b"BM") {
        return Some(ImageKind::Bmp);
    }
    if header.starts_with(b"II*\0") || header.starts_with(b"MM\0*") {
        return Some(ImageKind::Tiff);
    }
    // ISOBMFF (AVIF / HEIC / HEIF): a `ftyp` box at offset 4, the major brand at offset 8, then a
    // list of compatible brands. Some encoders only declare the image brand in the compatible list,
    // so scan every brand we read — not just the major one.
    if header.len() >= 12 && &header[4..8] == b"ftyp" {
        let declared = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
        // Cap the scan at the ftyp box size when it is sane, else at what we actually read.
        let limit = if declared >= 16 {
            declared.min(header.len())
        } else {
            header.len()
        };
        let mut brands: Vec<&[u8]> = vec![&header[8..12]];
        let mut offset = 16;
        while offset + 4 <= limit {
            brands.push(&header[offset..offset + 4]);
            offset += 4;
        }
        let has = |needle: &[u8; 4]| brands.contains(&needle.as_slice());
        if has(b"avif") || has(b"avis") {
            return Some(ImageKind::Avif);
        }
        if has(b"heic")
            || has(b"heix")
            || has(b"heim")
            || has(b"heis")
            || has(b"hevc")
            || has(b"hevx")
            || has(b"heif")
            || has(b"mif1")
            || has(b"msf1")
        {
            return Some(ImageKind::Heif);
        }
        // Some other ISOBMFF stream (e.g. an mp4/mov video brand) — not an image we transcode.
        return None;
    }
    None
}

/// Sniff the format of a file by reading its leading bytes. `None` on an unreadable file or an
/// unrecognized format.
pub fn sniff_image_kind_at(path: &Path) -> Option<ImageKind> {
    let mut header = [0u8; 32];
    let mut file = std::fs::File::open(path).ok()?;
    let read = std::io::Read::read(&mut file, &mut header).ok()?;
    sniff_image_kind(&header[..read])
}

/// Read an image's pixel dimensions from its header **without a full decode** (sc-6531,
/// Dataset Doctor). Used at dataset import to populate `TrainingDatasetItem.width/height`,
/// the foundation every Tier-0 quality check (min-resolution, crop-loss) relies on.
///
/// Header-only by design: `imagesize` parses just the format header, so this stays cheap and
/// — like the rest of this module — pulls in no native image-codec build. Returns `None` for
/// an unreadable file or an unrecognized format (the caller leaves dimensions absent rather
/// than failing the import).
pub fn image_dimensions(path: &Path) -> Option<(u32, u32)> {
    let size = imagesize::size(path).ok()?;
    Some((
        u32::try_from(size.width).ok()?,
        u32::try_from(size.height).ok()?,
    ))
}

/// SHA-256 of a file's raw bytes, lowercase hex (sc-6531). The stable content identity of a
/// stored dataset image: the exact-duplicate key for Tier-0 and the cache key that
/// invalidates a Tier-1 embedding exactly when the image bytes change (epic 6529 §6.3).
/// Streams the file so it never holds a whole image in memory.
pub fn file_content_hash(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    use std::io::Read as _;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    // SHA-256 is always 32 bytes → 64 lowercase-hex chars.
    let mut hex = String::with_capacity(64);
    for byte in hasher.finalize() {
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

/// Failure converting an image to PNG.
#[derive(Debug)]
pub struct TranscodeError(pub String);

impl fmt::Display for TranscodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for TranscodeError {}

/// Transcode any decoder-supported image at `src` to a lossless PNG at `dst`. For animated/burst
/// inputs (animated AVIF/GIF, HEIC bursts) the primary/first frame is taken.
///
/// macOS uses ImageIO-backed `sips` (always present) and falls back to `ffmpeg`; off macOS it uses
/// `ffmpeg` (resolved via `SCENEWORKS_FFMPEG`, else `ffmpeg` on PATH — the same binary the worker's
/// video path uses).
pub fn transcode_to_png(src: &Path, dst: &Path) -> Result<(), TranscodeError> {
    // Scene-linear HDR needs an explicit tone-map, not a platform decoder (sc-18790). Both
    // platform paths below merely CLIP at linear 1.0 — and they disagree about what they clip
    // (`sips` sRGB-encodes first, ffmpeg writes near-raw linear), so the same EXR previews
    // differently on macOS and Linux and every highlight above diffuse white collapses onto one
    // flat value. Route it to the explicit mapper instead.
    if sniff_image_kind_at(src).is_some_and(|kind| kind.png_transcode_is_lossy()) {
        return transcode_hdr_to_png(src, dst);
    }
    #[cfg(target_os = "macos")]
    {
        match run_sips_to_png(src, dst) {
            Ok(()) => Ok(()),
            Err(sips_error) => {
                // sips refused the format (rare); try ffmpeg if it is reachable before giving up.
                match run_ffmpeg_to_png(src, dst) {
                    Ok(()) => Ok(()),
                    Err(_) => Err(sips_error),
                }
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        run_ffmpeg_to_png(src, dst)
    }
}

/// Reinhard tone-map an unbounded scene-linear channel into display-referred `[0, 1]`.
///
/// `x / (1 + x)` is monotonic over the whole positive range: it maps 0 to 0, diffuse white (1.0)
/// to 0.5, and approaches — never reaches — 1.0. That is the property a preview needs and clipping
/// destroys: two different super-white values stay *distinguishable* instead of both becoming 255.
fn reinhard(x: f32) -> f32 {
    let x = if x.is_finite() { x.max(0.0) } else { 0.0 };
    x / (1.0 + x)
}

/// sRGB OETF (IEC 61966-2-1), linear `[0, 1]` → display-encoded `[0, 1]`.
fn linear_to_srgb(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// Tone-map a scene-linear HDR image to a display-referred PNG preview (sc-18790).
///
/// ffmpeg is used **only to decode** — every mapping decision is made here, in Rust, so the
/// derivative is byte-identical on macOS and Linux. That is the point: the platform decoders
/// disagree about how they flatten HDR, and a preview that differs by host is a preview you cannot
/// reason about.
///
/// The pipeline is: decode to planar `f32` → per-channel Reinhard → sRGB OETF → 8-bit RGB → PNG.
/// Per-channel (rather than luminance-based) Reinhard can desaturate a strongly-coloured highlight
/// slightly; it is chosen because it is monotonic per channel, which is exactly what keeps
/// super-white values apart in the output.
pub fn transcode_hdr_to_png(src: &Path, dst: &Path) -> Result<(), TranscodeError> {
    let (width, height) = hdr_dimensions(src)?;
    let pixels = hdr_pixel_count(width, height)?;
    let mut command = upload_ffmpeg_command();
    command
        .args(["-max_pixels", &MAX_HDR_PIXELS.to_string(), "-i"])
        .arg(src)
        .args([
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gbrpf32le",
            "pipe:1",
        ]);
    let output = run_upload_command(command, move |stdout| read_hdr_planes(stdout, pixels))?;
    require_success(&output, "decode the HDR image")?;
    let rgb = output.stdout;
    let buffer = image::RgbImage::from_raw(width, height, rgb)
        .ok_or_else(|| TranscodeError("tone-mapped buffer did not match the image size".into()))?;
    buffer
        .save_with_format(dst, image::ImageFormat::Png)
        .map_err(|error| TranscodeError(format!("could not write the HDR preview PNG: {error}")))?;
    ensure_nonempty_output(dst)
}

/// Bound HDR decoding before any frame allocation (64 megapixels, at most 192 MiB
/// for the RGB preview). Decoder-side `max_pixels` also constrains a header mismatch.
const MAX_HDR_PIXELS: usize = 64 * 1024 * 1024;
const MAX_HDR_DIMENSION: u32 = 16_384;

fn hdr_pixel_count(width: u32, height: u32) -> Result<usize, TranscodeError> {
    let pixels = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)));
    let Some(pixels) = pixels.filter(|pixels| *pixels > 0 && *pixels <= MAX_HDR_PIXELS) else {
        return Err(TranscodeError(format!(
            "HDR dimensions {width}x{height} exceed the {MAX_HDR_PIXELS}-pixel preview limit"
        )));
    };
    if width > MAX_HDR_DIMENSION
        || height > MAX_HDR_DIMENSION
        || pixels
            .checked_mul(3)
            .and_then(|n| n.checked_mul(4))
            .is_none()
    {
        return Err(TranscodeError(format!(
            "HDR dimensions {width}x{height} exceed the preview dimension/byte limit"
        )));
    }
    Ok(pixels)
}

/// EXR's dataWindow describes stored pixels, while FFmpeg emits the displayWindow
/// canvas (including black padding around a crop). imagesize reports only dataWindow.
/// Inspect both before decoding so a small crop cannot hide an unbounded canvas.
fn hdr_dimensions(src: &Path) -> Result<(u32, u32), TranscodeError> {
    use std::io::{Read, Seek, SeekFrom};
    if sniff_image_kind_at(src) != Some(ImageKind::OpenExr) {
        return image_dimensions(src).ok_or_else(|| {
            TranscodeError(format!(
                "could not read HDR image dimensions from {}",
                src.display()
            ))
        });
    }
    let file = std::fs::File::open(src).map_err(exr_header_error)?;
    let file_len = file.metadata().map_err(exr_header_error)?.len();
    let mut reader = std::io::BufReader::new(file);
    let mut prelude = [0u8; 8];
    reader.read_exact(&mut prelude).map_err(exr_header_error)?;
    let flags = u32::from_le_bytes(prelude[4..8].try_into().expect("four-byte version"));
    let name_limit = if flags & 0x400 != 0 { 255 } else { 31 };
    let mut data_window = None;
    let mut display_window = None;
    for _ in 0..4096 {
        let name = exr_attribute_name(&mut reader, name_limit)?;
        if name.is_empty() {
            return match (data_window, display_window) {
                (Some(_), Some(display)) => Ok(display),
                _ => Err(TranscodeError(
                    "OpenEXR is missing dataWindow/displayWindow dimensions".into(),
                )),
            };
        }
        let kind = exr_attribute_name(&mut reader, name_limit)?;
        let mut size = [0u8; 4];
        reader.read_exact(&mut size).map_err(exr_header_error)?;
        let size = u32::from_le_bytes(size);
        let end = reader
            .stream_position()
            .map_err(exr_header_error)?
            .checked_add(u64::from(size))
            .filter(|end| *end <= file_len)
            .ok_or_else(|| TranscodeError("OpenEXR attribute exceeds the file bounds".into()))?;
        if name == b"dataWindow" || name == b"displayWindow" {
            if kind != b"box2i" || size != 16 {
                return Err(TranscodeError(
                    "OpenEXR dimensions require a sixteen-byte box2i".into(),
                ));
            }
            let mut window = [0u8; 16];
            reader.read_exact(&mut window).map_err(exr_header_error)?;
            let coords: Vec<_> = window
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| i64::from(i32::from_le_bytes(*bytes)))
                .collect();
            let width = u32::try_from(coords[2] - coords[0] + 1).unwrap_or(0);
            let height = u32::try_from(coords[3] - coords[1] + 1).unwrap_or(0);
            hdr_pixel_count(width, height)?;
            let destination = if name == b"dataWindow" {
                &mut data_window
            } else {
                &mut display_window
            };
            if destination.replace((width, height)).is_some() {
                return Err(TranscodeError(
                    "OpenEXR has duplicate dimension attributes".into(),
                ));
            }
        }
        reader
            .seek(SeekFrom::Start(end))
            .map_err(exr_header_error)?;
    }
    Err(TranscodeError(
        "OpenEXR header exceeds the 4096-attribute preview limit".into(),
    ))
}

fn exr_header_error(error: std::io::Error) -> TranscodeError {
    TranscodeError(format!("could not read OpenEXR dimensions: {error}"))
}

fn exr_attribute_name(
    reader: &mut impl std::io::Read,
    limit: usize,
) -> Result<Vec<u8>, TranscodeError> {
    let mut name = Vec::new();
    for _ in 0..=limit {
        let mut byte = [0u8; 1];
        reader.read_exact(&mut byte).map_err(exr_header_error)?;
        if byte[0] == 0 {
            return Ok(name);
        }
        if name.len() == limit {
            break;
        }
        name.push(byte[0]);
    }
    Err(TranscodeError(
        "OpenEXR attribute name exceeds the format limit".into(),
    ))
}

async fn read_hdr_planes(
    mut stdout: impl tokio::io::AsyncRead + Unpin,
    pixels: usize,
) -> std::io::Result<Vec<u8>> {
    // Keep only the final RGB8 image and a fixed-size float chunk, never three full float planes.
    let bytes = pixels
        .checked_mul(3)
        .ok_or_else(|| std::io::Error::other("HDR RGB size overflow"))?;
    let mut rgb = Vec::new();
    rgb.try_reserve_exact(bytes).map_err(|error| {
        std::io::Error::other(format!("could not allocate HDR preview: {error}"))
    })?;
    rgb.resize(bytes, 0);
    let mut floats = [0u8; 16 * 1024];
    for channel in [1usize, 2, 0] {
        // gbrpf32le is G, B, R.
        let mut offset = 0;
        while offset < pixels {
            let count = (pixels - offset).min(floats.len() / 4);
            stdout.read_exact(&mut floats[..count * 4]).await?;
            for (index, sample) in floats[..count * 4].as_chunks::<4>().0.iter().enumerate() {
                let linear = f32::from_le_bytes(*sample);
                let mapped = linear_to_srgb(reinhard(linear));
                rgb[(offset + index) * 3 + channel] =
                    (mapped * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
            offset += count;
        }
    }
    if stdout.read(&mut [0u8; 1]).await? != 0 {
        return Err(std::io::Error::other(
            "HDR decoder returned more than one dimension-matched frame",
        ));
    }
    Ok(rgb)
}

/// Upload decodes are short utility work; operator settings may lower the bound, but
/// a long-running export setting cannot turn an upload into a twelve-hour request.
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(120);

fn upload_ffmpeg_command() -> tokio::process::Command {
    let mut command = tokio::process::Command::new(media_process::ffmpeg_program("ffmpeg"));
    command.args([
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file,pipe",
    ]);
    command
}

fn run_upload_command<F, R, T>(
    command: tokio::process::Command,
    read_stdout: F,
) -> Result<ProcessOutput<T>, TranscodeError>
where
    F: FnOnce(tokio::process::ChildStdout) -> R + Send,
    R: std::future::Future<Output = std::io::Result<T>> + Send + 'static,
    T: Send + 'static,
{
    // Public store/normalization APIs are synchronous and are also called inside Tokio
    // request handlers. A dedicated thread avoids nested Runtime::block_on panics.
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("media-utility".into())
            .spawn_scoped(scope, move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| {
                        TranscodeError(format!("could not start media utility runtime: {e}"))
                    })?;
                runtime
                    .block_on(media_process::run_command(
                        command,
                        None,
                        media_process::ffmpeg_execution_timeout().min(UPLOAD_TIMEOUT),
                        std::future::pending::<std::convert::Infallible>(),
                        read_stdout,
                    ))
                    .map_err(|e| TranscodeError(e.to_string()))
            })
            .map_err(|e| TranscodeError(format!("could not start media utility thread: {e}")))?
            .join()
            .map_err(|_| TranscodeError("media utility thread panicked".into()))?
    })
}

fn require_success<T>(output: &ProcessOutput<T>, action: &str) -> Result<(), TranscodeError> {
    match output.failure_detail() {
        None => Ok(()),
        Some(detail) => Err(TranscodeError(format!(
            "ffmpeg failed to {action}: {detail}"
        ))),
    }
}

#[cfg(target_os = "macos")]
fn run_sips_to_png(src: &Path, dst: &Path) -> Result<(), TranscodeError> {
    let mut command = tokio::process::Command::new("/usr/bin/sips");
    command
        .args(["-s", "format", "png"])
        .arg(src)
        .arg("--out")
        .arg(dst);
    let output = run_upload_command(command, media_process::discard_stdout)?;
    require_success(&output, "convert image to PNG with sips")?;
    ensure_nonempty_output(dst)
}

// Compatibility export for callers of the original transcoder API.
pub use crate::media_process::resolve_ffmpeg_program;

fn run_ffmpeg_to_png(src: &Path, dst: &Path) -> Result<(), TranscodeError> {
    let mut command = upload_ffmpeg_command();
    command
        .arg("-i")
        .arg(src)
        .args(["-frames:v", "1", "-update", "1"])
        .arg(dst);
    let output = run_upload_command(command, media_process::discard_stdout)?;
    require_success(&output, "convert image to PNG")?;
    ensure_nonempty_output(dst)
}

/// Transcode any decoder-supported audio at `src` to a canonical **PCM signed-16 little-endian
/// RIFF/WAVE** file at `dst`, preserving the source's sample rate and channel layout.
///
/// # Why the canonical form, and not "store what the user uploaded" (sc-18650)
///
/// PCM-16 RIFF/WAVE is not a preference, it is the ONLY audio SceneWorks can read back.
/// `sceneworks_worker::audio_jobs::read_wav_pcm16` is the single decoder in the product — it serves
/// the video reference-audio conditioning, the voice-clone tone-color reference, and the music
/// extend/edit source clip — and it refuses anything that is not RIFF/WAVE (`is not a RIFF/WAVE
/// file`) or not 16-bit PCM (`must be PCM 16-bit`). Storing an `.mp3` / `.flac` / `.m4a` verbatim
/// would produce an asset that LOOKS usable everywhere (the picker's `assetCanRenderAsAudio` keys on
/// `type == "audio"`, the browser plays it, the grid tiles it) and then fails at render time, deep
/// inside a job the user already queued. Converting at the door makes an imported clip
/// interchangeable with an Audio Studio render, which is the only other way an audio asset exists.
///
/// The rate and layout are deliberately NOT normalized: they are the caller's own signal, every
/// consumer reads them off the file, and the one consumer with a fixed-rate contract (MiniMax-H3's
/// 32 kHz audio VAE) resamples per request at conditioning time.
///
/// ffmpeg-only on every platform — `sips` is an image tool, so unlike [`transcode_to_png`] there is
/// no always-present macOS fallback. A host with no reachable ffmpeg cannot import audio that is
/// not already canonical, and says so.
///
/// # The one input that needs no ffmpeg (sc-22715)
///
/// A file whose header ALREADY declares exactly the encoding `read_wav_pcm16` accepts — RIFF/WAVE,
/// a `fmt ` chunk saying PCM (format tag 1) at 16 bits with a real rate and channel count, and a
/// non-empty `data` chunk — is copied byte for byte instead of being decoded and re-encoded to the
/// same thing. The normalization still ALWAYS runs (there is no pass-through at the store: every
/// audio upload comes through here), so the sc-18650 invariant that the stored file is the one
/// encoding the product reads back is kept; what is skipped is a decode→encode round trip whose
/// output would be sample-identical to its input. The check is [`is_canonical_pcm16_wav`], and it
/// is deliberately the same acceptance rule as the reader's, so an input that would satisfy the
/// reader is stored unchanged and everything else — 24-bit, float, WAVE_FORMAT_EXTENSIBLE, a
/// container with a picture in it — goes through ffmpeg exactly as before.
pub fn transcode_to_wav_pcm16(src: &Path, dst: &Path) -> Result<(), TranscodeError> {
    if is_canonical_pcm16_wav(src) {
        std::fs::copy(src, dst).map_err(|error| {
            TranscodeError(format!(
                "failed to copy the canonical PCM-16 WAV {} into place: {error}",
                src.display()
            ))
        })?;
        return ensure_nonempty(dst, "WAV");
    }
    let input = audio_input_policy(src)?;
    let mut command = upload_ffmpeg_command();
    command.args(["-format_whitelist", SAFE_AUDIO_DEMUXERS]);
    if let Some(demuxer) = input {
        command.args(["-f", demuxer]);
    }
    command
        .arg("-i")
        .arg(src)
        .args(["-map", "0:a:0", "-vn", "-c:a", "pcm_s16le"])
        .arg(dst);
    let output = run_upload_command(command, media_process::discard_stdout)?;
    require_success(&output, "convert audio to PCM-16 WAV")?;
    ensure_nonempty(dst, "WAV")
}

/// Whether `path` is a RIFF/WAVE file whose header declares exactly the encoding
/// `sceneworks_worker::audio_jobs::read_wav_pcm16` accepts: a `fmt ` chunk (>= 16 bytes) with
/// format tag 1 (PCM), 16 bits per sample, a non-zero rate and channel count, and a `data` chunk
/// with a non-zero declared length. Header-only, bounded to the first 64 KiB, and `false` on any
/// doubt — a chunk list that runs past the window without reaching `data`, a second `fmt ` after
/// the first, a truncated header — so the only file this ever admits is one ffmpeg would have
/// re-encoded to the same bytes.
pub fn is_canonical_pcm16_wav(path: &Path) -> bool {
    const WINDOW: usize = 64 * 1024;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut head = vec![0u8; WINDOW];
    let mut filled = 0usize;
    loop {
        match std::io::Read::read(&mut file, &mut head[filled..]) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(_) => return false,
        }
        if filled == head.len() {
            break;
        }
    }
    let bytes = &head[..filled];
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return false;
    }
    let le16 = |offset: usize| u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
    let le32 = |offset: usize| {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    };
    let mut position = 12usize;
    let mut format: Option<(u16, u16, u32, u16)> = None; // (tag, channels, rate, bits)
    let mut data_len: Option<u32> = None;
    while position + 8 <= bytes.len() {
        let id = &bytes[position..position + 4];
        let size = le32(position + 4) as usize;
        let body = position + 8;
        if id == b"fmt " {
            if format.is_some() || size < 16 || body + 16 > bytes.len() {
                return false;
            }
            format = Some((le16(body), le16(body + 2), le32(body + 4), le16(body + 14)));
        } else if id == b"data" {
            data_len = Some(le32(position + 4));
            break;
        }
        position = body + size + (size & 1);
    }
    let (Some((tag, channels, rate, bits)), Some(data_len)) = (format, data_len) else {
        return false;
    };
    tag == 1 && bits == 16 && channels > 0 && rate > 0 && data_len > 0
}

/// Binary audio/container demuxers, deliberately excluding all playlists and text
/// scripts (HLS, DASH, concat, SDP, image2, tty, etc.). Container uploads with cover
/// art/video remain supported: only the first audio stream is selected by the caller.
const SAFE_AUDIO_DEMUXERS: &str = "wav,mp3,flac,ogg,mov,matroska,webm,avi,flv,live_flv,mxf,aac,aiff,asf,amr,caf,au,ac3,eac3,dts,dtshd,mpeg,mpegts,ape,wv,tta,tak,alac,mlp,truehd,oma,dsf,iff,w64,rf64,loas,mpc,mpc8,shorten,spdif,adp,adx,aea,afc,aix,alp,aptx,aptx_hd,ast,avs,bfi,brstm,dcstr,dss,dxa,ea,ea_cdata,epaf,fsb,g722,g723_1,g726,g726le,gsm,ircam,iss,iv8,kvag,luodat,lvf,lxf,mca,mgsts,msf,mtaf,musx,nistsphere,nsp,nsv,nut,psxstr,pvf,qcp,r3d,rm,rpl,rsd,rso,sbc,sds,sdx,sfx,sln,smacker,smjpeg,sndstr,sunrast,svag,swf,thp,txd,vag,voc,vpk,vqf,wady,wsaud,wsd,xmv,xvag,xwma";

/// Recognize normal formats from bytes rather than the untrusted name/MIME. For
/// other binary codecs allow FFmpeg's bounded probe, confined to binary demuxers.
/// Text never reaches a decoder, even when named .mp3 or prefixed with a BOM.
fn audio_input_policy(path: &Path) -> Result<Option<&'static str>, TranscodeError> {
    use std::io::Read;
    let mut header = [0u8; 4096];
    let mut file = std::fs::File::open(path)
        .map_err(|e| TranscodeError(format!("could not inspect uploaded audio: {e}")))?;
    let count = file
        .read(&mut header)
        .map_err(|e| TranscodeError(format!("could not inspect uploaded audio: {e}")))?;
    let h = &header[..count];
    // ID3 may precede MP3, AAC, or FLAC; probe only the safe binary demuxers.
    if h.starts_with(b"ID3") {
        return Ok(None);
    }
    let format = if h.len() >= 12
        && matches!(&h[..4], b"RIFF" | b"RF64" | b"RIFX")
        && &h[8..12] == b"WAVE"
    {
        Some("wav")
    } else if h.starts_with(b"fLaC") {
        Some("flac")
    } else if h.starts_with(b"OggS") {
        Some("ogg")
    } else if h.len() >= 2 && h[0] == 0xff && h[1] & 0xe0 == 0xe0 && h[1] & 0x06 != 0 {
        Some("mp3")
    } else if h.len() >= 2 && h[0] == 0xff && h[1] & 0xf6 == 0xf0 {
        Some("aac")
    } else if h.len() >= 12 && &h[..4] == b"FORM" && matches!(&h[8..12], b"AIFF" | b"AIFC") {
        Some("aiff")
    } else if h.len() >= 8
        && matches!(
            &h[4..8],
            b"ftyp" | b"moov" | b"mdat" | b"free" | b"wide" | b"skip"
        )
    {
        Some("mov")
    } else if h.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        Some("matroska")
    } else if h.starts_with(&[0x30, 0x26, 0xb2, 0x75, 0x8e, 0x66, 0xcf, 0x11]) {
        Some("asf")
    } else if h.starts_with(b"#!AMR") {
        Some("amr")
    } else if h.starts_with(b"caff") {
        Some("caf")
    } else if h.starts_with(b".snd") {
        Some("au")
    } else {
        None
    };
    if format.is_none()
        && (h.is_empty()
            || std::str::from_utf8(h).is_ok_and(|text| {
                text.chars()
                    .all(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t'))
            })
            || h.starts_with(&[0xff, 0xfe])
            || h.starts_with(&[0xfe, 0xff]))
    {
        return Err(TranscodeError("uploaded audio is text or an unrecognized audio header; playlists and external references are not supported".into()));
    }
    Ok(format)
}

fn ensure_nonempty_output(dst: &Path) -> Result<(), TranscodeError> {
    ensure_nonempty(dst, "PNG")
}

fn ensure_nonempty(dst: &Path, label: &str) -> Result<(), TranscodeError> {
    match std::fs::metadata(dst) {
        Ok(meta) if meta.len() > 0 => Ok(()),
        Ok(_) => Err(TranscodeError(format!(
            "transcode produced an empty {label}"
        ))),
        Err(error) => Err(TranscodeError(format!(
            "transcode produced no {label} output: {error}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_audio_playlists_and_text_even_with_audio_names() {
        let dir = tempfile::tempdir().unwrap();
        for (index, payload) in [
            "#EXTM3U\nhttp://127.0.0.1:9/private.wav\n",
            "[playlist]\nFile1=file:///etc/passwd\n",
            "ffconcat version 1.0\nfile '/etc/passwd'\n",
            "<MPD><BaseURL>http://localhost/private</BaseURL></MPD>",
            "v=0\no=- 0 0 IN IP4 127.0.0.1\n",
            "\u{feff}#EXTM3U\nsecret.wav\n",
            "",
            "ordinary text",
        ]
        .iter()
        .enumerate()
        {
            let src = dir.path().join(format!("hostile-{index}.mp3"));
            std::fs::write(&src, payload).unwrap();
            let dst = dir.path().join(format!("out-{index}.wav"));
            let error = transcode_to_wav_pcm16(&src, &dst).unwrap_err();
            assert!(error.to_string().contains("playlists"), "{error}");
            assert!(!dst.exists());
        }
        // A fake binary prefix cannot bypass the decoder policy: these names
        // never appear in the whitelist, even for a binary-looking payload.
        for blocked in ["hls", "dash", "concat", "sdp", "image2", "tty", "libgme"] {
            assert!(!SAFE_AUDIO_DEMUXERS.split(',').any(|name| name == blocked));
        }
    }

    #[test]
    fn audio_sniff_preserves_normal_container_families_and_ignores_extensions() {
        let dir = tempfile::tempdir().unwrap();
        for (index, (magic, expected)) in [
            (b"RIFF0000WAVE".as_slice(), Some("wav")),
            (b"RF64ffffWAVE", Some("wav")),
            (b"fLaC", Some("flac")),
            (b"OggS", Some("ogg")),
            (b"ID3", None),
            (&[0xff, 0xfb, 0x90, 0x00], Some("mp3")),
            (&[0xff, 0xf1, 0x50, 0x00], Some("aac")),
            (b"FORM0000AIFF", Some("aiff")),
            (b"FORM0000AIFC", Some("aiff")),
            (b"0000ftypM4A ", Some("mov")),
            (b"0000moov0000", Some("mov")),
            (&[0x1a, 0x45, 0xdf, 0xa3], Some("matroska")),
            (
                &[0x30, 0x26, 0xb2, 0x75, 0x8e, 0x66, 0xcf, 0x11],
                Some("asf"),
            ),
            (b"#!AMR\n", Some("amr")),
            (b"caff", Some("caf")),
            (b".snd", Some("au")),
            (&[0x80, 0x00, 0x81, 0x00], None),
        ]
        .into_iter()
        .enumerate()
        {
            let src = dir.path().join(format!("not-audio-{index}.txt"));
            std::fs::write(&src, magic).unwrap();
            assert_eq!(
                audio_input_policy(&src).unwrap(),
                expected,
                "fixture {index}"
            );
        }
    }

    #[test]
    fn upload_ffmpeg_has_noninteractive_local_protocol_policy() {
        let command = upload_ffmpeg_command();
        let args: Vec<_> = command
            .as_std()
            .get_args()
            .map(|a| a.to_str().unwrap())
            .collect();
        assert!(args.contains(&"-nostdin"));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["-protocol_whitelist", "file,pipe"]));
        assert!(UPLOAD_TIMEOUT > Duration::ZERO && UPLOAD_TIMEOUT <= Duration::from_secs(120));
    }

    #[test]
    #[ignore = "subprocess fixture for synchronous utility bridge"]
    fn utility_process_fixture() {
        eprintln!("utility-bridge-complete");
    }

    fn assert_sync_utility_bridge() {
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command.args([
            "--exact",
            "media_convert::tests::utility_process_fixture",
            "--ignored",
            "--nocapture",
        ]);
        let output = run_upload_command(command, media_process::discard_stdout).unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("utility-bridge-complete"));
    }

    #[test]
    fn synchronous_utility_bridge_works_outside_a_runtime() {
        assert_sync_utility_bridge();
    }

    #[tokio::test]
    async fn synchronous_utility_bridge_works_inside_an_existing_runtime() {
        assert_sync_utility_bridge();
    }

    #[test]
    fn hdr_dimensions_reject_zero_overflow_and_oversize_before_spawn_or_allocation() {
        for (w, h) in [
            (0, 1),
            (1, 0),
            (u32::MAX, u32::MAX),
            (8193, 8192),
            (16_385, 1),
        ] {
            assert!(hdr_pixel_count(w, h).is_err(), "{w}x{h}");
        }
        assert_eq!(hdr_pixel_count(8192, 8192).unwrap(), MAX_HDR_PIXELS);
        assert_eq!(hdr_pixel_count(4096, 2160).unwrap(), 4096 * 2160);
        let dir = tempfile::tempdir().unwrap();
        let exr = dir.path().join("huge.exr");
        let dst = dir.path().join("out.png");
        let mut header = vec![0x76, 0x2f, 0x31, 0x01, 2, 0, 0, 0];
        header.extend_from_slice(b"dataWindow\0box2i\0");
        header.extend_from_slice(&16u32.to_le_bytes());
        for coord in [0i32, 0, 100_000, 100_000] {
            header.extend_from_slice(&coord.to_le_bytes());
        }
        std::fs::write(&exr, header).unwrap();
        assert_eq!(image_dimensions(&exr), Some((100_001, 100_001)));
        let error = transcode_hdr_to_png(&exr, &dst).unwrap_err();
        assert!(error.to_string().contains("preview limit"));
        assert!(!dst.exists());
    }

    #[test]
    fn exr_dimension_headers_reject_duplicate_and_out_of_bounds_attributes() {
        fn window(header: &mut Vec<u8>, name: &[u8]) {
            header.extend_from_slice(name);
            header.extend_from_slice(b"\0box2i\0");
            header.extend_from_slice(&16u32.to_le_bytes());
            header.extend_from_slice(&[0u8; 16]); // one pixel at the origin
        }
        let dir = tempfile::tempdir().unwrap();
        let mut valid = vec![0x76, 0x2f, 0x31, 0x01, 2, 0, 0, 0];
        window(&mut valid, b"dataWindow");
        window(&mut valid, b"displayWindow");
        let mut duplicate = valid.clone();
        window(&mut duplicate, b"displayWindow");
        duplicate.push(0);
        let mut out_of_bounds = valid;
        out_of_bounds.extend_from_slice(b"notes\0string\0");
        out_of_bounds.extend_from_slice(&u32::MAX.to_le_bytes());
        for (name, header, message) in [
            ("duplicate", duplicate, "duplicate"),
            ("bounds", out_of_bounds, "file bounds"),
            (
                "truncated",
                vec![0x76, 0x2f, 0x31, 0x01, 2, 0, 0, 0],
                "could not read",
            ),
        ] {
            let src = dir.path().join(format!("{name}.exr"));
            std::fs::write(&src, header).unwrap();
            let error = hdr_dimensions(&src).unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    fn reachable_test_ffmpeg() -> Option<String> {
        let program = media_process::ffmpeg_program("ffmpeg");
        match Command::new(&program).arg("-version").output() {
            Ok(output) if output.status.success() => Some(program),
            other => {
                assert!(
                    std::env::var("SCENEWORKS_REQUIRE_FFMPEG").as_deref() != Ok("1"),
                    "required ffmpeg is unavailable: {program} ({other:?})"
                );
                eprintln!("SKIPPED media codec fixtures: {program} unavailable ({other:?})");
                None
            }
        }
    }

    #[test]
    fn normal_audio_formats_import_as_pcm16_with_rate_and_channels_preserved() {
        let Some(ffmpeg) = reachable_test_ffmpeg() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        for (extension, codec) in [
            ("wav", "pcm_s24le"),
            ("wav", "pcm_f32le"),
            ("mp3", "libmp3lame"),
            ("flac", "flac"),
            ("ogg", "vorbis"),
            ("opus", "libopus"),
            ("aac", "aac"),
            ("m4a", "aac"),
            ("mp4", "aac"),
            ("mov", "pcm_s16le"),
            ("mkv", "pcm_s16le"),
            ("webm", "libopus"),
            ("aiff", "pcm_s16be"),
            ("caf", "pcm_s16le"),
            ("au", "pcm_s16be"),
            ("wma", "wmav2"),
        ] {
            let src = dir.path().join(format!("{codec}.{extension}"));
            let built = Command::new(&ffmpeg)
                .args([
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000",
                    "-t",
                    "0.05",
                    "-ac",
                    "2",
                    "-c:a",
                    codec,
                    "-strict",
                    "-2",
                    "-y",
                ])
                .arg(&src)
                .output()
                .unwrap();
            if !built.status.success() {
                let message = String::from_utf8_lossy(&built.stderr);
                if message.contains("Unknown encoder") || message.contains("Encoder not found") {
                    eprintln!("SKIPPED optional {extension}/{codec} fixture: {message}");
                    continue;
                }
                panic!("could not build {extension}/{codec}: {message}");
            }
            let dst = dir
                .path()
                .join(format!("{codec}-{extension}.normalized.wav"));
            transcode_to_wav_pcm16(&src, &dst)
                .unwrap_or_else(|error| panic!("{extension}/{codec}: {error}"));
            assert!(is_canonical_pcm16_wav(&dst), "{extension}/{codec}");
            if codec == "pcm_f32le" {
                let async_dst = dir.path().join("from-async-handler.wav");
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async { transcode_to_wav_pcm16(&src, &async_dst) })
                    .unwrap();
                assert!(is_canonical_pcm16_wav(&async_dst));
            }
            let bytes = std::fs::read(&dst).unwrap();
            let fmt = bytes
                .windows(4)
                .position(|window| window == b"fmt ")
                .unwrap()
                + 8;
            assert_eq!(
                u16::from_le_bytes(bytes[fmt + 2..fmt + 4].try_into().unwrap()),
                2,
                "{extension}/{codec} channels"
            );
            assert_eq!(
                u32::from_le_bytes(bytes[fmt + 4..fmt + 8].try_into().unwrap()),
                48000,
                "{extension}/{codec} rate"
            );
        }
    }

    #[test]
    fn audio_import_keeps_first_audio_stream_and_ignores_cover_art() {
        let Some(ffmpeg) = reachable_test_ffmpeg() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("cover.mp3");
        let dst = dir.path().join("cover.wav");
        let built = Command::new(&ffmpeg)
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=sample_rate=44100",
                "-f",
                "lavfi",
                "-i",
                "color=red:size=16x16:rate=1",
                "-map",
                "0:a:0",
                "-map",
                "1:v:0",
                "-c:a",
                "libmp3lame",
                "-c:v",
                "png",
                "-disposition:v",
                "attached_pic",
                "-t",
                "0.05",
                "-y",
            ])
            .arg(&src)
            .output()
            .unwrap();
        if !built.status.success()
            && String::from_utf8_lossy(&built.stderr).contains("Unknown encoder")
        {
            eprintln!(
                "SKIPPED optional cover-art fixture: {}",
                String::from_utf8_lossy(&built.stderr)
            );
            return;
        }
        assert!(
            built.status.success(),
            "could not build cover-art fixture: {}",
            String::from_utf8_lossy(&built.stderr)
        );
        transcode_to_wav_pcm16(&src, &dst).unwrap();
        assert!(is_canonical_pcm16_wav(&dst));
        let bytes = std::fs::read(&dst).unwrap();
        let fmt = bytes
            .windows(4)
            .position(|window| window == b"fmt ")
            .unwrap()
            + 8;
        assert_eq!(
            u16::from_le_bytes(bytes[fmt + 2..fmt + 4].try_into().unwrap()),
            1
        );
        assert_eq!(
            u32::from_le_bytes(bytes[fmt + 4..fmt + 8].try_into().unwrap()),
            44100
        );
    }

    #[test]
    fn binary_disguised_playlists_and_remote_protocols_cannot_import_audio() {
        let Some(_) = reachable_test_ffmpeg() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        for (index, payload) in [
            b"ID3\x04\0\0\0\0\0\0#EXTM3U\n#EXT-X-TARGETDURATION:1\nhttp://127.0.0.1:9/private.wav\n".as_slice(),
            b"ID3\x04\0\0\0\0\0\0ffconcat version 1.0\nfile '/etc/passwd'\n",
            b"\xff\x80#EXTM3U\nfile:///etc/passwd\n",
            b"\xff\x80[playlist]\nFile1=http://127.0.0.1:9/private.wav\n",
        ].into_iter().enumerate() {
            let src = dir.path().join(format!("binary-playlist-{index}.mp3"));
            let dst = dir.path().join(format!("out-{index}.wav"));
            std::fs::write(&src, payload).unwrap();
            let error = transcode_to_wav_pcm16(&src, &dst).unwrap_err();
            let message = error.to_string();
            assert!(!message.contains("Connection refused") && !message.contains("Failed to open segment"), "decoder attempted a remote reference: {message}");
            assert!(!dst.exists() || std::fs::metadata(&dst).unwrap().len() == 0);
        }
        // A recognized container still cannot open a network protocol, even if a
        // future caller constructs its input URL directly rather than using a file.
        let mut command = upload_ffmpeg_command();
        command.args(["-i", "http://127.0.0.1:9/private.wav", "-f", "null", "-"]);
        let output = run_upload_command(command, media_process::discard_stdout).unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("not on whitelist"));
    }

    #[test]
    fn upload_rejects_a_local_playlist_that_unrestricted_ffmpeg_would_follow() {
        let Some(ffmpeg) = reachable_test_ffmpeg() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let private = dir.path().join("private.wav");
        let playlist = dir.path().join("disguised.mp3");
        let control = dir.path().join("unrestricted.wav");
        let guarded = dir.path().join("guarded.wav");
        std::fs::write(&private, wav_bytes(1, 16, 1, 800, &[])).unwrap();
        std::fs::write(&playlist, "ffconcat version 1.0\nfile 'private.wav'\n").unwrap();
        let decoded = Command::new(&ffmpeg)
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(&playlist)
            .args(["-map", "0:a:0", "-c:a", "pcm_s16le", "-y"])
            .arg(&control)
            .output()
            .unwrap();
        assert!(
            decoded.status.success(),
            "unrestricted ffmpeg control must demonstrate local indirection: {}",
            String::from_utf8_lossy(&decoded.stderr)
        );
        assert!(is_canonical_pcm16_wav(&control));
        let error = transcode_to_wav_pcm16(&playlist, &guarded).unwrap_err();
        assert!(error.to_string().contains("playlists"), "{error}");
        assert!(!guarded.exists());
    }

    #[test]
    fn ffmpeg_image_fallback_preserves_normal_raster_formats() {
        let Some(ffmpeg) = reachable_test_ffmpeg() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        for extension in ["png", "jpg", "webp", "bmp", "gif", "tiff", "avif"] {
            let src = dir.path().join(format!("source.{extension}"));
            if extension == "webp" {
                image::RgbImage::from_pixel(16, 12, image::Rgb([20, 80, 140]))
                    .save_with_format(&src, image::ImageFormat::WebP)
                    .unwrap();
            } else {
                let built = Command::new(&ffmpeg)
                    .args([
                        "-nostdin",
                        "-v",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "testsrc=size=16x12:rate=1",
                        "-frames:v",
                        "1",
                        "-y",
                    ])
                    .arg(&src)
                    .output()
                    .unwrap();
                if !built.status.success() {
                    let message = String::from_utf8_lossy(&built.stderr);
                    if message.contains("Encoder") || message.contains("encoder") {
                        eprintln!("SKIPPED optional {extension} encoder fixture: {message}");
                        continue;
                    }
                    panic!("could not build {extension}: {message}");
                }
            }
            let dst = dir.path().join(format!("from-{extension}.png"));
            run_ffmpeg_to_png(&src, &dst).unwrap_or_else(|error| panic!("{extension}: {error}"));
            let image = image::open(&dst).unwrap();
            assert_eq!((image.width(), image.height()), (16, 12));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn platform_decoder_preserves_heic_import() {
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("source.png");
        let heic = dir.path().join("source.heic");
        let out = dir.path().join("preview.png");
        image::RgbImage::from_pixel(64, 64, image::Rgb([20, 80, 140]))
            .save(&png)
            .unwrap();
        let built = Command::new("/usr/bin/sips")
            .args(["-s", "format", "heic"])
            .arg(&png)
            .arg("--out")
            .arg(&heic)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "could not build HEIC: {}",
            String::from_utf8_lossy(&built.stderr)
        );
        assert_eq!(sniff_image_kind_at(&heic), Some(ImageKind::Heif));
        transcode_to_png(&heic, &out).unwrap();
        let image = image::open(&out).unwrap();
        assert_eq!((image.width(), image.height()), (64, 64));
    }

    #[tokio::test]
    async fn hdr_plane_reader_checks_truncation_extra_bytes_and_channel_order() {
        let mut planes = Vec::new();
        for value in [1.0f32, 2.0, 4.0] {
            planes.extend_from_slice(&value.to_le_bytes());
        }
        let rgb = read_hdr_planes(planes.as_slice(), 1).await.unwrap();
        let mapped = |v| (linear_to_srgb(reinhard(v)) * 255.0 + 0.5) as u8;
        assert_eq!(rgb, [mapped(4.0), mapped(1.0), mapped(2.0)]);
        assert!(read_hdr_planes(&planes[..11], 1).await.is_err());
        planes.push(0);
        assert!(read_hdr_planes(planes.as_slice(), 1)
            .await
            .unwrap_err()
            .to_string()
            .contains("more than one"));
        // Cross the fixed float-chunk boundary and preserve per-plane RGB order.
        let pixels = 4097;
        let mut planes = Vec::new();
        for value in [f32::NAN, -1.0, 8.0] {
            for _ in 0..pixels {
                planes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let rgb = read_hdr_planes(planes.as_slice(), pixels).await.unwrap();
        assert_eq!(rgb.len(), pixels * 3);
        assert!(rgb
            .as_chunks::<3>()
            .0
            .iter()
            .all(|pixel| *pixel == [mapped(8.0), 0, 0]));
    }

    /// A RIFF/WAVE with one `fmt ` chunk of the given tag/bits and a `data` chunk of `samples`
    /// 16-bit frames, with `extra` chunks inserted before `fmt `.
    fn wav_bytes(tag: u16, bits: u16, channels: u16, samples: usize, extra: &[u8]) -> Vec<u8> {
        let block_align = channels * bits / 8;
        let data_len = samples * usize::from(block_align);
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((4 + extra.len() + 24 + 8 + data_len) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(extra);
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&tag.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&8_000u32.to_le_bytes());
        wav.extend_from_slice(&(8_000 * u32::from(block_align)).to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_len as u32).to_le_bytes());
        for index in 0..data_len {
            wav.push((index % 251) as u8);
        }
        wav
    }

    /// sc-22715: a file that already IS the one encoding the product reads back is copied through
    /// without an ffmpeg — byte for byte — and everything that is not (the float WAV sc-18650 was
    /// about, 24-bit, an extensible header, a file with no data) still needs the converter.
    #[test]
    fn a_canonical_pcm16_wav_is_copied_through_and_nothing_else_is() {
        let dir = tempfile::tempdir().expect("temp dir");
        let canonical = dir.path().join("canonical.wav");
        std::fs::write(&canonical, wav_bytes(1, 16, 1, 800, &[])).unwrap();
        assert!(is_canonical_pcm16_wav(&canonical));
        let out = dir.path().join("out.wav");
        transcode_to_wav_pcm16(&canonical, &out).expect("a canonical WAV needs no ffmpeg");
        assert_eq!(
            std::fs::read(&out).unwrap(),
            std::fs::read(&canonical).unwrap(),
            "the copy is byte-identical"
        );

        // A LIST chunk ahead of `fmt ` is still canonical: the walker skips it.
        let mut list = Vec::from(*b"LIST");
        list.extend_from_slice(&5u32.to_le_bytes());
        list.extend_from_slice(b"INFOx");
        list.push(0); // word-align the odd body
        let listed = dir.path().join("listed.wav");
        std::fs::write(&listed, wav_bytes(1, 16, 2, 800, &list)).unwrap();
        assert!(is_canonical_pcm16_wav(&listed));

        for (label, bytes) in [
            ("IEEE float", wav_bytes(3, 32, 1, 800, &[])),
            ("24-bit PCM", wav_bytes(1, 24, 1, 800, &[])),
            ("WAVE_FORMAT_EXTENSIBLE", wav_bytes(0xFFFE, 16, 2, 800, &[])),
            ("zero channels", wav_bytes(1, 16, 0, 0, &[])),
            ("empty data", wav_bytes(1, 16, 1, 0, &[])),
            ("not a WAV", b"RIFF\0\0\0\0WEBPVP8 ".to_vec()),
        ] {
            let path = dir.path().join(format!("{}.wav", label.replace(' ', "_")));
            std::fs::write(&path, bytes).unwrap();
            assert!(
                !is_canonical_pcm16_wav(&path),
                "{label} must go through ffmpeg, not be copied through"
            );
        }
        // A second `fmt ` chunk is a file the reader would read differently from the header this
        // walker saw, so it is refused the copy and handed to ffmpeg.
        let mut second_fmt = Vec::from(*b"fmt ");
        second_fmt.extend_from_slice(&16u32.to_le_bytes());
        second_fmt.extend_from_slice(&[0u8; 16]);
        let doubled = dir.path().join("doubled.wav");
        std::fs::write(&doubled, wav_bytes(1, 16, 1, 800, &second_fmt)).unwrap();
        assert!(!is_canonical_pcm16_wav(&doubled));
    }

    #[test]
    fn sniffs_natively_supported_formats() {
        let png = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
        assert_eq!(sniff_image_kind(&png), Some(ImageKind::Png));
        assert!(sniff_image_kind(&png).unwrap().is_natively_supported());

        let jpeg = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(sniff_image_kind(&jpeg), Some(ImageKind::Jpeg));

        let mut webp = Vec::from(*b"RIFF");
        webp.extend_from_slice(&[0, 0, 0, 0]);
        webp.extend_from_slice(b"WEBP");
        assert_eq!(sniff_image_kind(&webp), Some(ImageKind::WebP));
    }

    #[test]
    fn sniffs_unsupported_raster_formats_as_needing_transcode() {
        assert_eq!(sniff_image_kind(b"GIF89a....."), Some(ImageKind::Gif));
        assert_eq!(sniff_image_kind(b"BM......"), Some(ImageKind::Bmp));
        assert_eq!(sniff_image_kind(b"II*\0...."), Some(ImageKind::Tiff));
        assert_eq!(sniff_image_kind(b"MM\0*...."), Some(ImageKind::Tiff));
        for kind in [ImageKind::Gif, ImageKind::Bmp, ImageKind::Tiff] {
            assert!(!kind.is_natively_supported());
        }
    }

    /// OpenEXR is recognized by magic, is not browser-renderable, and — unlike every other
    /// transcoded format — is flagged lossy-to-transcode so import keeps the original bytes.
    ///
    /// The last assertion is the load-bearing one for sc-18790: if OpenEXR ever joined the
    /// convert-and-discard set, an HDR frame would be silently replaced by an 8-bit tone-mapped
    /// PNG and the scene-linear original would be unrecoverable.
    #[test]
    fn sniffs_openexr_and_marks_it_lossy_to_transcode() {
        // EXR magic number 20000630 (0x01312F76), little-endian, then the version field.
        let exr = [0x76u8, 0x2F, 0x31, 0x01, 0x02, 0x00, 0x00, 0x00];
        assert_eq!(sniff_image_kind(&exr), Some(ImageKind::OpenExr));
        assert!(
            !ImageKind::OpenExr.is_natively_supported(),
            "no browser decodes OpenEXR — it must never be treated as directly renderable"
        );
        assert!(
            ImageKind::OpenExr.png_transcode_is_lossy(),
            "OpenEXR must be stored as-is; a PNG rendition is a proxy, not a substitute"
        );
        assert_eq!(ImageKind::OpenExr.canonical(), ("exr", "image/x-exr"));

        // Every OTHER transcoded format is lossless to convert, so import may keep only the PNG.
        // Asserting this keeps the flag a real distinction rather than a field nothing reads.
        for kind in [
            ImageKind::Gif,
            ImageKind::Bmp,
            ImageKind::Tiff,
            ImageKind::Avif,
            ImageKind::Heif,
        ] {
            assert!(
                !kind.png_transcode_is_lossy(),
                "{} is 8-bit and converts losslessly to PNG",
                kind.label()
            );
        }
    }

    /// EXTERNAL — ffmpeg renders a real OpenEXR file to PNG through the shared transcoder, which is
    /// what the `?thumbnail=` derivative route relies on to preview an HDR asset.
    ///
    /// Uses ffmpeg to BUILD the fixture as well as to consume it, so no binary blob is checked in
    /// (this repo synthesizes image fixtures in code rather than committing them). Skips loudly
    /// when ffmpeg is unreachable.
    #[test]
    fn transcodes_a_real_openexr_to_png_via_the_platform_decoder() {
        let ffmpeg = std::env::var("SCENEWORKS_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_owned());
        match Command::new(&ffmpeg).arg("-version").output() {
            Ok(out) if out.status.success() => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!(
                    "SKIPPED (external): transcodes_a_real_openexr_to_png_via_the_platform_decoder \
                     — `{ffmpeg}` is not reachable."
                );
                return;
            }
            other => panic!("failed to probe {ffmpeg}: {other:?}"),
        }

        // A self-removing root: the guard IS the cleanup, so it stays bound for the whole test.
        // A PID-named tree under env::temp_dir() would survive a panic and collide with a sibling.
        let dir = tempfile::tempdir().expect("temp dir");
        let exr = dir.path().join("frame.exr");
        let png = dir.path().join("frame.png");

        let built = Command::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi"])
            .args(["-i", "testsrc=size=32x32:rate=1", "-frames:v", "1"])
            .args(["-pix_fmt", "gbrpf32le", "-y"])
            .arg(&exr)
            .output()
            .expect("run ffmpeg");
        assert!(
            built.status.success(),
            "could not build the EXR fixture: {}",
            String::from_utf8_lossy(&built.stderr)
        );

        // The bytes ffmpeg wrote must be what our own sniffer calls an OpenEXR.
        let header = std::fs::read(&exr).expect("read exr");
        assert_eq!(
            sniff_image_kind(&header[..32.min(header.len())]),
            Some(ImageKind::OpenExr),
            "a real ffmpeg-written EXR must sniff as OpenEXR"
        );

        transcode_to_png(&exr, &png).expect("transcode EXR to PNG");
        let rendered = std::fs::read(&png).expect("read png");
        assert_eq!(
            sniff_image_kind(&rendered[..32.min(rendered.len())]),
            Some(ImageKind::Png),
            "the preview derivative must be a real PNG"
        );
    }

    #[test]
    fn hdr_preview_checks_and_preserves_cropped_exr_display_windows() {
        let Some(ffmpeg) = reachable_test_ffmpeg() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let original = dir.path().join("original.exr");
        let built = Command::new(&ffmpeg)
            .args([
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=33x17:rate=1",
                "-frames:v",
                "1",
                "-pix_fmt",
                "gbrpf32le",
                "-y",
            ])
            .arg(&original)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let bytes = std::fs::read(&original).unwrap();
        let marker = b"displayWindow\0box2i\0";
        let window = bytes
            .windows(marker.len())
            .position(|part| part == marker)
            .unwrap()
            + marker.len()
            + 4;
        for (name, max, allowed) in [("cropped", 63i32, true), ("huge-canvas", 100_000, false)] {
            let mut bytes = bytes.clone();
            for (index, coord) in [0i32, 0, max, max].into_iter().enumerate() {
                bytes[window + index * 4..window + (index + 1) * 4]
                    .copy_from_slice(&coord.to_le_bytes());
            }
            let src = dir.path().join(format!("{name}.exr"));
            let dst = dir.path().join(format!("{name}.png"));
            std::fs::write(&src, bytes).unwrap();
            assert_eq!(
                image_dimensions(&src),
                Some((33, 17)),
                "imagesize exposes only the stored crop"
            );
            if allowed {
                assert_eq!(hdr_dimensions(&src).unwrap(), (64, 64));
                transcode_hdr_to_png(&src, &dst).unwrap();
                let image = image::open(&dst).unwrap().to_rgb8();
                assert_eq!(image.dimensions(), (64, 64));
                assert_eq!(image.get_pixel(63, 63).0, [0, 0, 0]);
            } else {
                let error = transcode_hdr_to_png(&src, &dst).unwrap_err();
                assert!(error.to_string().contains("preview limit"), "{error}");
                assert!(!dst.exists());
            }
        }
    }

    /// The tone-map curve itself: highlights above diffuse white stay below 255 AND stay apart.
    ///
    /// Pure-Rust, so it runs everywhere and pins the property clipping destroys. A clipping
    /// preview maps every super-white value onto the same 255, which is what made the previous
    /// "tone-mapped proxy" label untrue.
    #[test]
    fn tone_map_keeps_highlights_below_clipping_and_distinguishable() {
        let code = |linear: f32| (linear_to_srgb(reinhard(linear)) * 255.0 + 0.5) as u8;

        assert_eq!(code(0.0), 0, "black must stay black");
        // Diffuse white is mapped WELL below 255, which is what reserves the headroom.
        let white = code(1.0);
        assert!(
            (150..=220).contains(&white),
            "diffuse white should sit mid-high, got {white}"
        );

        // Every highlight above diffuse white stays below the clipping point...
        let ladder: Vec<u8> = [2.0f32, 4.0, 8.0, 16.0, 64.0]
            .iter()
            .map(|v| code(*v))
            .collect();
        for (linear, got) in [2.0f32, 4.0, 8.0, 16.0, 64.0].iter().zip(&ladder) {
            assert!(
                *got < 255,
                "linear {linear} clipped to {got} — highlights must not saturate"
            );
        }
        // ...and stays strictly ordered, so two different super-white values remain readable.
        for pair in ladder.windows(2) {
            assert!(
                pair[1] > pair[0],
                "super-white values collapsed onto the same code: {ladder:?}"
            );
        }
    }

    /// EXTERNAL — an end-to-end HDR preview: ffmpeg builds a real EXR carrying a 0 → 8.0 ramp, our
    /// mapper renders it, and the decoded PNG must show the super-white end unclipped and ordered.
    ///
    /// The unit test above pins the curve; this pins the whole path (decode → map → encode),
    /// which is where a plane-order or stride mistake would hide.
    #[test]
    fn hdr_preview_renders_a_super_white_ramp_without_clipping() {
        let ffmpeg = std::env::var("SCENEWORKS_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_owned());
        match Command::new(&ffmpeg).arg("-version").output() {
            Ok(out) if out.status.success() => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!(
                    "SKIPPED (external): hdr_preview_renders_a_super_white_ramp_without_clipping \
                     — `{ffmpeg}` is not reachable."
                );
                return;
            }
            other => panic!("failed to probe {ffmpeg}: {other:?}"),
        }
        let dir = tempfile::tempdir().expect("temp dir");
        let exr = dir.path().join("ramp.exr");
        let png = dir.path().join("ramp.png");

        // A horizontal 0 → 8.0 linear ramp: `geq` writes raw values, so the right-hand half is
        // genuinely above diffuse white rather than merely bright.
        let built = Command::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi"])
            .args(["-i", "color=c=black:s=64x8:r=1", "-frames:v", "1"])
            .args(["-vf", "format=gbrpf32le,geq=r='8*X/W':g='8*X/W':b='8*X/W'"])
            .args(["-y"])
            .arg(&exr)
            .output()
            .expect("run ffmpeg");
        assert!(
            built.status.success(),
            "could not build the HDR ramp fixture: {}",
            String::from_utf8_lossy(&built.stderr)
        );

        transcode_hdr_to_png(&exr, &png).expect("tone-map the HDR ramp");
        let decoded = image::open(&png).expect("open preview").to_rgb8();
        assert_eq!(decoded.dimensions(), (64, 8));

        let row: Vec<u8> = (0..64u32).map(|x| decoded.get_pixel(x, 0)[0]).collect();
        assert!(
            *row.last().unwrap() < 255,
            "the 8.0 end of the ramp clipped to 255: {:?}",
            &row[56..]
        );
        // Sample points well inside the super-white region must differ from each other.
        let (mid, high) = (row[40], row[62]);
        assert!(
            high > mid,
            "super-white samples collapsed: linear ~5 -> {mid}, linear ~7.75 -> {high}"
        );
    }

    #[test]
    fn sniffs_avif_and_heic_isobmff_brands() {
        // size(4) + "ftyp" + major brand + minor + one compatible brand
        let avif = isobmff(b"avif", &[b"mif1"]);
        assert_eq!(sniff_image_kind(&avif), Some(ImageKind::Avif));

        let heic = isobmff(b"heic", &[b"mif1", b"heic"]);
        assert_eq!(sniff_image_kind(&heic), Some(ImageKind::Heif));

        // HEIC declared only via a compatible brand (major brand is the generic mif1).
        let heic_compat = isobmff(b"mif1", &[b"heic"]);
        assert_eq!(sniff_image_kind(&heic_compat), Some(ImageKind::Heif));

        for kind in [ImageKind::Avif, ImageKind::Heif] {
            assert!(!kind.is_natively_supported());
        }
    }

    #[test]
    fn ignores_non_image_isobmff_and_garbage() {
        // A plain mp4 video brand is ISOBMFF but not an image we transcode.
        let mp4 = isobmff(b"isom", &[b"mp42", b"isom"]);
        assert_eq!(sniff_image_kind(&mp4), None);
        assert_eq!(sniff_image_kind(b""), None);
        assert_eq!(sniff_image_kind(&[0u8; 16]), None);
    }

    /// Build a minimal ISOBMFF header: `size(4) ftyp major minor compatible...`.
    fn isobmff(major: &[u8; 4], compatible: &[&[u8; 4]]) -> Vec<u8> {
        let size = 16 + compatible.len() * 4;
        let mut bytes = (size as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(b"ftyp");
        bytes.extend_from_slice(major);
        bytes.extend_from_slice(&[0, 0, 0, 0]); // minor version
        for brand in compatible {
            bytes.extend_from_slice(*brand);
        }
        bytes
    }

    // sc-6531 (Dataset Doctor) — header-only dimensions + content hash.

    #[test]
    fn image_dimensions_reads_header_without_decode() {
        let dir = tempfile::tempdir().expect("temp dir");

        // Two formats whose dimensions sit at fixed header offsets, so a header-only reader needs
        // no pixel decode. Both are non-square, so a width/height transposition would be caught.
        let gif = dir.path().join("wide.gif");
        std::fs::write(&gif, gif_header(5, 3)).expect("write gif");
        assert_eq!(image_dimensions(&gif), Some((5, 3)));

        let png = dir.path().join("wide.png");
        std::fs::write(&png, png_header(4, 2)).expect("write png");
        assert_eq!(image_dimensions(&png), Some((4, 2)));
    }

    #[test]
    fn image_dimensions_returns_none_for_non_image_or_missing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, b"not an image").expect("write file");
        assert_eq!(image_dimensions(&path), None);
        assert_eq!(image_dimensions(&dir.path().join("missing.png")), None);
    }

    #[test]
    fn file_content_hash_is_sha256_of_bytes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("blob.bin");
        std::fs::write(&path, b"hello").expect("write file");
        // Known SHA-256("hello"); identical bytes must hash identically (the exact-duplicate key).
        assert_eq!(
            file_content_hash(&path).expect("hash"),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    /// Minimal PNG: 8-byte signature + a complete IHDR chunk carrying `width`/`height` — enough
    /// for a header-only dimension read (no IDAT; the reader does not verify the CRC).
    fn png_header(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        bytes.extend_from_slice(&13u32.to_be_bytes()); // IHDR data length
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[0x08, 0x06, 0x00, 0x00, 0x00]); // depth, color, compression, filter, interlace
        bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // CRC placeholder
        bytes
    }

    /// Minimal GIF89a: 6-byte signature + the logical-screen `width`/`height` (LE u16) — the
    /// fixed-offset header a dimension read needs (no color table / image data required).
    fn gif_header(width: u16, height: u16) -> Vec<u8> {
        let mut bytes = Vec::from(*b"GIF89a");
        bytes.extend_from_slice(&width.to_le_bytes());
        bytes.extend_from_slice(&height.to_le_bytes());
        bytes.extend_from_slice(&[0x00, 0x00, 0x00]); // packed fields, bg color, aspect ratio
        bytes
    }

    /// Real conversion path: a hand-rolled BMP → PNG via `sips`. macOS-only because `sips` is the
    /// always-present decoder there; the ffmpeg path is exercised by the worker integration tests.
    #[cfg(target_os = "macos")]
    #[test]
    fn transcodes_bmp_to_png_via_sips() {
        let dir = tempfile::tempdir().expect("temp dir");
        let src = dir.path().join("pixel.bmp");
        let dst = dir.path().join("pixel.png");
        std::fs::write(&src, one_pixel_bmp()).expect("write bmp");

        transcode_to_png(&src, &dst).expect("sips transcodes BMP to PNG");

        let out = std::fs::read(&dst).expect("read png");
        assert_eq!(sniff_image_kind(&out), Some(ImageKind::Png));
    }

    /// A valid 1×1 24-bit BMP (no Rust image dep needed to build one).
    #[cfg(target_os = "macos")]
    fn one_pixel_bmp() -> Vec<u8> {
        let mut bytes = Vec::new();
        // BITMAPFILEHEADER (14 bytes)
        bytes.extend_from_slice(b"BM");
        bytes.extend_from_slice(&58u32.to_le_bytes()); // file size
        bytes.extend_from_slice(&0u16.to_le_bytes()); // reserved1
        bytes.extend_from_slice(&0u16.to_le_bytes()); // reserved2
        bytes.extend_from_slice(&54u32.to_le_bytes()); // pixel data offset
                                                       // BITMAPINFOHEADER (40 bytes)
        bytes.extend_from_slice(&40u32.to_le_bytes()); // header size
        bytes.extend_from_slice(&1i32.to_le_bytes()); // width
        bytes.extend_from_slice(&1i32.to_le_bytes()); // height
        bytes.extend_from_slice(&1u16.to_le_bytes()); // planes
        bytes.extend_from_slice(&24u16.to_le_bytes()); // bpp
        bytes.extend_from_slice(&0u32.to_le_bytes()); // compression (BI_RGB)
        bytes.extend_from_slice(&0u32.to_le_bytes()); // image size
        bytes.extend_from_slice(&2835i32.to_le_bytes()); // x ppm
        bytes.extend_from_slice(&2835i32.to_le_bytes()); // y ppm
        bytes.extend_from_slice(&0u32.to_le_bytes()); // colors used
        bytes.extend_from_slice(&0u32.to_le_bytes()); // important colors
                                                      // Pixel data: one BGR pixel + row padding to a 4-byte boundary.
        bytes.extend_from_slice(&[0x20, 0x40, 0x80, 0x00]);
        bytes
    }
}
