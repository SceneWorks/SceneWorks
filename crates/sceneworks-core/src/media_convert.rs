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

use std::borrow::Cow;
use std::fmt;
use std::path::Path;
#[cfg(any(windows, test))]
use std::path::PathBuf;
use std::process::Command;

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
}

impl ImageKind {
    /// True when the worker's `png`/`jpeg`/`webp` `image` build can decode this directly; everything
    /// else must be transcoded to PNG first.
    pub fn is_natively_supported(self) -> bool {
        matches!(self, ImageKind::Png | ImageKind::Jpeg | ImageKind::WebP)
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

#[cfg(target_os = "macos")]
fn run_sips_to_png(src: &Path, dst: &Path) -> Result<(), TranscodeError> {
    let output = Command::new("/usr/bin/sips")
        .arg("-s")
        .arg("format")
        .arg("png")
        .arg(src)
        .arg("--out")
        .arg(dst)
        .output()
        .map_err(|error| TranscodeError(format!("failed to run sips: {error}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(TranscodeError(format!(
            "sips failed to convert image to PNG: {}",
            stderr.trim()
        )));
    }
    ensure_nonempty_output(dst)
}

/// Resolve an ffmpeg program string to a spawn-ready path, working around a Windows launch
/// failure. An executable reached through a symlink/reparse point — e.g. WinGet's
/// `…\Microsoft\WinGet\Links\ffmpeg.exe` shim, a symbolic link to the real Gyan.FFmpeg binary —
/// cannot be started under redirection-trust enforcement: `CreateProcess` returns
/// `ERROR_UNTRUSTED_MOUNT_POINT` (os error 448), the very wall the HF-cache symlinks hit in the
/// model loaders (see `sceneworks_worker::model_jobs`). We locate the program (searching `PATH`
/// for a bare name) and, if it is a reparse point, resolve it to its real target — read **without
/// traversing it** (`read_link`, chasing chained links) — and return that plain-file path. A plain
/// executable, a name we cannot locate, or any non-Windows host returns the input unchanged so
/// normal resolution (PATH search, the host's own ordering) is preserved.
///
/// This is the single resolver shared by the worker's `media_jobs::run_ffmpeg` and the transcoder
/// below, so every ffmpeg spawn — timeline export, video encode, audio mux, frame extract, image
/// transcode — is reparse-safe.
pub fn resolve_ffmpeg_program(program: &str) -> Cow<'_, str> {
    #[cfg(windows)]
    {
        if let Some(real) = resolve_ffmpeg_program_reparse(program) {
            return Cow::Owned(real);
        }
    }
    Cow::Borrowed(program)
}

/// Windows reparse-resolution core, factored out so it is unit-testable off-Windows
/// (`#[cfg(any(windows, test))]`); the public resolver only invokes it under `#[cfg(windows)]`.
/// Returns `Some(real_path)` only when the program resolves to a reparse point that must be
/// rewritten; `None` (keep the original string) for a plain file or a name we cannot locate.
#[cfg(any(windows, test))]
fn resolve_ffmpeg_program_reparse(program: &str) -> Option<String> {
    let located = locate_executable(program)?;
    // A plain file launches fine — leave it to normal resolution.
    if !is_symlink(&located) {
        return None;
    }
    let real = resolve_reparse_chain(&located)?;
    Some(real.to_string_lossy().into_owned())
}

/// `symlink_metadata` does not traverse the link, so a 448-prone reparse point is still detected.
#[cfg(any(windows, test))]
fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_symlink())
        .unwrap_or(false)
}

/// Find the file `CreateProcess` would launch. An explicit path is used as-is; a bare name is
/// searched on `PATH` (trying the name and a `.exe` suffix). A candidate is accepted if it is a
/// file *or* a symlink — the latter via `symlink_metadata`, which does not traverse and so still
/// sees a link whose target cannot be opened.
#[cfg(any(windows, test))]
fn locate_executable(program: &str) -> Option<PathBuf> {
    let direct = Path::new(program);
    if direct.is_absolute() || direct.components().count() > 1 {
        return (direct.exists() || is_symlink(direct)).then(|| direct.to_path_buf());
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        for suffix in ["", ".exe"] {
            let candidate = dir.join(format!("{program}{suffix}"));
            if candidate.is_file() || is_symlink(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

/// Follow a symlink to the first non-symlink path, resolving relative targets against the link's
/// parent and bounding the walk so a cyclic link cannot loop forever. The caller only invokes this
/// once `start` is known to be a symlink, so `read_link` runs without ever traversing the reparse
/// point that would otherwise hit os error 448.
#[cfg(any(windows, test))]
fn resolve_reparse_chain(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    for _ in 0..16 {
        if !is_symlink(&current) {
            return Some(current);
        }
        let target = std::fs::read_link(&current).ok()?;
        current = if target.is_absolute() {
            target
        } else {
            current.parent()?.join(target)
        };
    }
    None
}

fn run_ffmpeg_to_png(src: &Path, dst: &Path) -> Result<(), TranscodeError> {
    // Mirror the worker's ffmpeg resolution: the desktop app points SCENEWORKS_FFMPEG at its bundled
    // binary (it ships no system ffmpeg); the server stack leaves it unset and uses PATH.
    let configured = std::env::var("SCENEWORKS_FFMPEG")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "ffmpeg".to_owned());
    let program = resolve_ffmpeg_program(&configured);
    let output = Command::new(program.as_ref())
        .arg("-y")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(src)
        // Take a single frame so an animated input collapses to one still PNG.
        .arg("-frames:v")
        .arg("1")
        .arg("-update")
        .arg("1")
        .arg(dst)
        .output()
        .map_err(|error| {
            TranscodeError(format!(
                "failed to run ffmpeg ({program}); ensure ffmpeg is installed: {error}"
            ))
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(TranscodeError(format!(
            "ffmpeg failed to convert image to PNG: {}",
            stderr.trim()
        )));
    }
    ensure_nonempty_output(dst)
}

fn ensure_nonempty_output(dst: &Path) -> Result<(), TranscodeError> {
    match std::fs::metadata(dst) {
        Ok(meta) if meta.len() > 0 => Ok(()),
        Ok(_) => Err(TranscodeError("transcode produced an empty PNG".to_owned())),
        Err(error) => Err(TranscodeError(format!(
            "transcode produced no PNG output: {error}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The ffmpeg reparse-resolution fix (os error 448): a program reached through a symlink — e.g.
    /// WinGet's `…\WinGet\Links\ffmpeg.exe` shim — resolves to its real, non-reparse target so
    /// `CreateProcess` never has to traverse the reparse point. Skips when the platform won't let
    /// the test create a symlink (e.g. Windows without Developer Mode).
    #[test]
    fn resolve_ffmpeg_program_follows_reparse_to_real_binary() {
        let dir = tempfile::tempdir().expect("temp dir");
        let real = dir.path().join("ffmpeg-real.exe");
        std::fs::write(&real, b"MZ-stub").expect("write real binary");
        let link = dir.path().join("ffmpeg.exe");

        // Absolute target, like WinGet's Links shim.
        #[cfg(unix)]
        let made = std::os::unix::fs::symlink(&real, &link).is_ok();
        #[cfg(windows)]
        let made = std::os::windows::fs::symlink_file(&real, &link).is_ok();
        #[cfg(not(any(unix, windows)))]
        let made = false;
        if !made {
            return; // no privilege to create a symlink — nothing to resolve
        }
        assert!(is_symlink(&link), "precondition: the shim is a symlink");

        let resolved = resolve_ffmpeg_program_reparse(&link.to_string_lossy())
            .expect("a symlinked program resolves to its target");
        assert_eq!(
            std::fs::canonicalize(&resolved).expect("canonicalize resolved"),
            std::fs::canonicalize(&real).expect("canonicalize real"),
            "resolved path must point at the real binary"
        );
        assert!(
            !is_symlink(Path::new(&resolved)),
            "resolved path must not itself be a reparse point"
        );

        // A plain binary is left untouched (None → caller keeps the original program).
        assert!(
            resolve_ffmpeg_program_reparse(&real.to_string_lossy()).is_none(),
            "a plain file needs no rewrite"
        );
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
