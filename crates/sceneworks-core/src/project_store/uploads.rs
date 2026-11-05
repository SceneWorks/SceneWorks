//! ProjectStore uploads aggregate.

use super::*;

/// Key inside an imported asset's top-level `extra` that carries the sanitized workflow envelope
/// the uploaded file had embedded (sc-15949).
///
/// Named `imported` on purpose. It says where the recipe came from — a file someone else's install
/// wrote — and keeps it distinguishable from this asset's own `recipe`, which for an upload is the
/// `mode: "upload"` stub and stays that way. sc-15952 reads this to offer "Use this recipe".
pub const IMPORTED_WORKFLOW_KEY: &str = "importedWorkflow";

/// The outcome of import-time image normalization (sc-6143): the path whose bytes should be moved
/// into the asset store, plus the mime type and file extension to record. When `transcoded_temp` is
/// set, the move consumes that temp PNG (not the original upload), so the caller must remove the
/// original source itself.
pub(super) struct NormalizedUpload {
    pub(super) source_path: PathBuf,
    pub(super) content_type: String,
    pub(super) extension: String,
    transcoded_temp: Option<PathBuf>,
    /// The sanitized workflow envelope the upload carried in a PNG text chunk, when a
    /// [`WorkflowScan::Read`] caller asked and one was found (sc-15949).
    ///
    /// Read from the ORIGINAL upload bytes, so it survives a transcode that re-encodes them.
    pub(super) workflow: Option<WorkflowShare>,
}

/// Whether an upload's metadata is searched for an embedded workflow envelope (sc-15949).
///
/// A parameter rather than an unconditional scan, so the cost is paid only where the answer is
/// used. `import_asset` records what it finds under `extra.importedWorkflow`; the training-dataset
/// lane has no recipe slot and no reader for one, so it opts out and pays nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WorkflowScan {
    /// Look for a `sceneworks:workflow` chunk, before any transcode.
    Read,
    /// Do not look.
    Skip,
}

/// Normalize a valid-but-unsupported image upload (AVIF/HEIC/HEIF/TIFF/BMP/GIF) to lossless PNG
/// before it is stored, so every downstream decode site, thumbnail, and preview stays on a format
/// it can read (sc-6143). The format is sniffed by content, never the extension, so a `.png` that is
/// really AVIF is still handled. Already-supported formats (png/jpeg/webp) and non-image uploads
/// pass through untouched — no temp file, no re-encode, no quality loss.
///
/// `scan` also makes this the one place an incoming image is searched for an embedded workflow
/// (sc-15949), because this is the one place that knows what is about to happen to the bytes; see
/// the comment at the scan itself for why the ordering is load-bearing in both directions.
pub(super) fn normalize_image_upload(
    source_path: &Path,
    content_type: &str,
    filename: &str,
    work_dir: &Path,
    scan: WorkflowScan,
) -> ProjectStoreResult<NormalizedUpload> {
    let passthrough = || NormalizedUpload {
        source_path: source_path.to_path_buf(),
        content_type: content_type.to_owned(),
        extension: upload_extension(filename, content_type),
        transcoded_temp: None,
        workflow: None,
    };
    if !content_type.starts_with("image/") {
        return Ok(passthrough());
    }
    let Some(kind) = crate::media_convert::sniff_image_kind_at(source_path) else {
        // Unrecognized magic (e.g. SVG) — leave it as-is; we can't transcode what we can't sniff.
        return Ok(passthrough());
    };
    // sc-15949: the chunk scan goes HERE, and the position is the decision.
    //
    // BEFORE the natively-supported branch below, because that branch is a `return`: a PNG — the
    // one format that carries our chunk — would otherwise never be looked at.
    //
    // BEFORE the transcode further down, because `sips`/`ffmpeg` re-encode from pixels and drop
    // every ancillary chunk by definition. A scan afterwards could only ever report an absence, so
    // a workflow found here is carried forward on `NormalizedUpload` even though the stored bytes
    // end up re-encoded.
    //
    // And it scans CHUNKS, not pixels. `read_workflow_chunk_file` stops at the first IDAT, and when
    // that finds nothing it walks the remaining framing with no output buffer, which `png` answers
    // by consuming the IDAT payload instead of inflating it. So this function's whole point — a
    // natively-decodable upload is stored byte-for-byte, with no decode anywhere on the path —
    // still holds.
    let workflow = match scan {
        WorkflowScan::Read => read_upload_workflow(source_path),
        WorkflowScan::Skip => None,
    };
    if kind.is_natively_supported() {
        // Already decodable: keep the bytes (no re-encode) but record the format we actually
        // detected, so a mislabeled extension (a `.avif` that is really PNG) is corrected.
        let (extension, mime) = kind.canonical();
        return Ok(NormalizedUpload {
            source_path: source_path.to_path_buf(),
            content_type: mime.to_owned(),
            extension: format!(".{extension}"),
            transcoded_temp: None,
            workflow,
        });
    }
    if kind.png_transcode_is_lossy() {
        // Scene-linear HDR (sc-18790). Storing the PNG and discarding the source is right for every
        // other transcoded format — their pixels survive the conversion — but here it would throw
        // away the float latitude the file exists for, leaving a tone-mapped proxy and no way back.
        // So the original bytes are stored; the browser-renderable PNG is produced on demand by the
        // `?thumbnail=` derivative route, which already falls back to `transcode_to_png` when the
        // `image` crate cannot decode a source. Download therefore serves the original.
        let (extension, mime) = kind.canonical();
        return Ok(NormalizedUpload {
            source_path: source_path.to_path_buf(),
            content_type: mime.to_owned(),
            extension: format!(".{extension}"),
            transcoded_temp: None,
            workflow,
        });
    }
    let temp_png = work_dir.join(format!("upload-transcode-{}.png", random_hex(8)?));
    crate::media_convert::transcode_to_png(source_path, &temp_png).map_err(|error| {
        let _ = fs::remove_file(&temp_png);
        ProjectStoreError::BadRequest(format!(
            "Could not convert {} image to a supported format: {error}",
            kind.label()
        ))
    })?;
    Ok(NormalizedUpload {
        source_path: temp_png.clone(),
        content_type: "image/png".to_owned(),
        extension: ".png".to_owned(),
        transcoded_temp: Some(temp_png),
        workflow,
    })
}

/// Normalize an audio upload to the canonical PCM-16 RIFF/WAVE the product can actually read back
/// (sc-18650), ALWAYS — there is no pass-through branch HERE.
///
/// The rule is deliberately unlike [`normalize_image_upload`]'s, where an already-decodable PNG or
/// JPEG is stored byte-for-byte. Audio has exactly one reader,
/// `sceneworks_worker::audio_jobs::read_wav_pcm16`, and it accepts exactly one encoding, so
/// "already supported" is a much narrower set than "already audio": a 24-bit WAV, a float WAV, or a
/// WAVE_FORMAT_EXTENSIBLE header are all `.wav` files it refuses. Every upload therefore goes
/// through `transcode_to_wav_pcm16`; the ONE encoding that converter copies through unchanged
/// (`media_convert::is_canonical_pcm16_wav`, sc-22715) is exactly the reader's own acceptance
/// rule, decided inside the converter rather than by a sniff here, so the stored file is the
/// canonical one on every path and a host with no ffmpeg can still import a file that already is.
///
/// A conversion failure is a `BadRequest`: the caller named a file that is not decodable audio (or
/// carries no audio stream), which is a fact about the upload, not about the host. The one host-shaped
/// failure — no reachable ffmpeg — is reported through the same error with ffmpeg's own message, which
/// says so.
fn normalize_audio_upload(
    source_path: &Path,
    work_dir: &Path,
) -> ProjectStoreResult<NormalizedUpload> {
    let temp_wav = work_dir.join(format!("upload-transcode-{}.wav", random_hex(8)?));
    crate::media_convert::transcode_to_wav_pcm16(source_path, &temp_wav).map_err(|error| {
        let _ = fs::remove_file(&temp_wav);
        ProjectStoreError::BadRequest(format!(
            "Could not convert the uploaded audio to a supported format: {error}"
        ))
    })?;
    Ok(NormalizedUpload {
        source_path: temp_wav.clone(),
        content_type: "audio/wav".to_owned(),
        extension: ".wav".to_owned(),
        transcoded_temp: Some(temp_wav),
        // An audio file carries no SceneWorks workflow chunk — that envelope is a PNG ancillary
        // chunk, and nothing writes an audio equivalent.
        workflow: None,
    })
}

/// The clip facts a stored PCM WAV declares in its own header: rate, channel count, and the duration
/// those two plus the `data` chunk length imply.
#[derive(Clone, Copy, Debug, PartialEq)]
struct WavFacts {
    sample_rate: u32,
    channels: u16,
    duration_secs: f64,
}

/// Read [`WavFacts`] out of a RIFF/WAVE header, or `None` if the file is not one this product wrote.
///
/// Header-only: it walks the chunk list for `fmt ` and `data` and reads the `data` chunk's declared
/// LENGTH, never its bytes, so a 200 MB clip costs one small read. It is intentionally NOT a second
/// `read_wav_pcm16` — that one lives in the worker, decodes samples, and answers a different
/// question. This one answers "what does the sidecar record about this file", which the store has to
/// answer without linking a decoder.
fn wav_facts(path: &Path) -> Option<WavFacts> {
    // The fixed prelude plus the largest offset read below (`fmt `'s 16-byte body).
    let mut head = vec![0u8; 4096];
    let read = {
        use std::io::Read;
        let mut file = fs::File::open(path).ok()?;
        let mut filled = 0usize;
        loop {
            match file.read(&mut head[filled..]) {
                Ok(0) => break,
                Ok(count) => filled += count,
                Err(_) => return None,
            }
            if filled == head.len() {
                break;
            }
        }
        filled
    };
    let bytes = &head[..read];
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
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
    let mut format: Option<(u16, u32, u16)> = None; // (channels, sample_rate, bits)
    let mut data_len: Option<u32> = None;
    while position + 8 <= bytes.len() {
        let id = &bytes[position..position + 4];
        let size = le32(position + 4);
        let body = position + 8;
        if id == b"fmt " && size >= 16 && body + 16 <= bytes.len() {
            format = Some((le16(body + 2), le32(body + 4), le16(body + 14)));
        } else if id == b"data" {
            data_len = Some(size);
            break;
        }
        // RIFF chunks are word-aligned: an odd body is followed by a pad byte.
        position = body + size as usize + (size as usize & 1);
    }
    let (channels, sample_rate, bits) = format?;
    if channels == 0 || sample_rate == 0 || bits == 0 {
        return None;
    }
    let frame_bytes = u32::from(channels) * u32::from(bits) / 8;
    let duration_secs = match (data_len, frame_bytes) {
        (Some(len), frame) if frame > 0 => f64::from(len / frame) / f64::from(sample_rate),
        _ => 0.0,
    };
    Some(WavFacts {
        sample_rate,
        channels,
        // Three decimals, the precision `build_render_asset` and the audio sidecar already record
        // durations at.
        duration_secs: (duration_secs * 1000.0).round() / 1000.0,
    })
}

/// Look for a SceneWorks workflow envelope in an upload's PNG metadata, degrading EVERY failure to
/// "no workflow found" (sc-15949).
///
/// This file came from a stranger, and it is the epic's main attack surface. The guards are
/// [`crate::workflow_png::read_workflow_chunk_file`]'s and are not re-implemented here: a bounded
/// metadata budget, a 1 MiB cap on the decompressed text so a zip bomb costs a megabyte, a refusal
/// of duplicate chunks and lying chunk lengths, and — the important one — a single parse path
/// through the sc-15946 allow-list, so the envelope is reduced at value granularity on the way IN
/// exactly as it was on the way out. Nothing here deserializes an `ImageJobRequest`, or anything
/// else, from the file.
///
/// Every error becomes `None`, deliberately and with no exception. The user asked to import an
/// image, and they get their image: a hostile or malformed chunk costs them the recipe field and
/// nothing else. `NotPng` is not even unusual — it is what every JPEG and WebP upload returns — so
/// the typed error is logged at debug rather than surfaced.
fn read_upload_workflow(source_path: &Path) -> Option<WorkflowShare> {
    match crate::workflow_png::read_workflow_chunk_file(source_path) {
        Ok(found) => found,
        Err(error) => {
            tracing::debug!(
                path = %source_path.display(),
                %error,
                "upload carries no readable workflow chunk; importing it as a plain asset"
            );
            None
        }
    }
}

/// Move a (possibly transcoded) upload into place, cleaning up the transcode temp on a move failure
/// and the now-orphaned original on success. Shared by the asset + training-dataset import paths.
pub(super) fn move_normalized_upload(
    normalized: &NormalizedUpload,
    original_source: &Path,
    media_path: &Path,
) -> ProjectStoreResult<()> {
    if let Err(error) = move_or_copy_file(&normalized.source_path, media_path) {
        if let Some(temp) = &normalized.transcoded_temp {
            let _ = fs::remove_file(temp);
        }
        return Err(error);
    }
    // A transcode moved the temp PNG into place, so the original upload was never consumed — drop it.
    if normalized.transcoded_temp.is_some() {
        let _ = fs::remove_file(original_source);
    }
    Ok(())
}

pub(super) fn normalize_asset_tags(tags: &[String]) -> ProjectStoreResult<Vec<String>> {
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::new();
    for tag in tags {
        let value = tag.trim().to_ascii_lowercase();
        if value.is_empty() {
            continue;
        }
        if value.len() > 40 {
            return Err(ProjectStoreError::BadRequest(
                "Asset tags must be 40 characters or fewer".to_owned(),
            ));
        }
        if seen.insert(value.clone()) {
            normalized.push(value);
        }
    }
    if normalized.len() > 24 {
        return Err(ProjectStoreError::BadRequest(
            "Assets can have at most 24 tags".to_owned(),
        ));
    }
    Ok(normalized)
}

pub(super) fn safe_filename(value: &str, fallback: &str) -> String {
    let name = value.replace('\\', "/");
    let stem = Path::new(name.rsplit('/').next().unwrap_or_default())
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let slug = slugify(stem, fallback, Some(64));
    if slug.is_empty() {
        fallback.to_owned()
    } else {
        slug
    }
}

/// The sidecar `type` an upload's mime maps to. `"audio"` since sc-18650 — the same string
/// `build_generated_asset_sidecar` dispatches on for an Audio Studio render, so an imported clip and
/// a rendered one are one asset kind rather than two.
fn media_type_for_mime(mime_type: &str) -> ProjectStoreResult<&'static str> {
    if mime_type == crate::tensor_artifact::MIME {
        return Ok("tensor");
    }
    if mime_type.starts_with("image/") {
        return Ok("image");
    }
    if mime_type.starts_with("video/") {
        return Ok("video");
    }
    if mime_type.starts_with("audio/") {
        return Ok("audio");
    }
    Err(ProjectStoreError::BadRequest(
        "Only image, video and audio uploads are supported".to_owned(),
    ))
}

pub(super) fn guess_mime_from_filename(filename: &str) -> Option<String> {
    if Path::new(filename)
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("npy"))
    {
        return Some(crate::tensor_artifact::MIME.to_owned());
    }
    mime_guess::from_path(filename)
        .first_raw()
        .map(str::to_owned)
        .or_else(|| {
            match Path::new(filename)
                .extension()
                .and_then(|value| value.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("heic") => Some("image/heic".to_owned()),
                Some("heif") => Some("image/heif".to_owned()),
                Some("avif") => Some("image/avif".to_owned()),
                Some("tif" | "tiff") => Some("image/tiff".to_owned()),
                // OpenEXR (sc-18790). `image/x-exr` is inert and starts with `image/`, which
                // is what admits it to SAFE_UPLOAD_EXTENSIONS and to the `?thumbnail=` route.
                Some("exr") => Some("image/x-exr".to_owned()),
                _ => None,
            }
        })
}

/// Extensions we allow a stored upload to carry, keyed by the media type they belong to. A stored
/// filename's extension re-derives the serve mime in [`ProjectStore::project_file`] via
/// [`guess_mime_from_filename`], and that mime is echoed verbatim into the file-serving endpoint's
/// `Content-Type` header, so an attacker who controls the stored extension controls the served
/// content type on the API origin. This allow-list is the gate: only extensions that map to an
/// inert media type (image/video) are ever stored, so a `video/mp4` upload named `evil.html` can
/// never be stored `.html` and served `text/html` → no content-type confusion / stored XSS
/// (sc-8872). Notably it excludes `.svg` (→ `image/svg+xml`, script-capable) and every
/// document/script extension.
///
/// Every extension here re-derives, through `guess_mime_from_filename`, to a mime that starts with
/// `image/`, `video/` or `audio/` and is NOT `image/svg+xml` — the property the tests pin.
///
/// `wav` is the only audio entry and that is not an oversight (sc-18650): `normalize_audio_upload`
/// converts EVERY audio upload to PCM-16 RIFF/WAVE, so `.wav` is the only extension an audio asset
/// is ever stored under. An `.mp3` is admitted at the door and stored `.wav`; the source extension
/// never reaches the filesystem, so it never needs to be on this list.
pub(super) const SAFE_UPLOAD_EXTENSIONS: &[&str] = &[
    // Raster images (SVG deliberately omitted — it is script-capable).
    "png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff", "avif", "heic", "heif",
    // Scene-linear HDR frames (sc-18790) — stored as-is, previewed as a derivative.
    "exr", // Video.
    "mp4", "m4v", "mov", "webm", "mkv", "avi", "ogv", "mpeg", "mpg", "wmv", "flv", "3gp",
    "3g2", // Audio: the single canonical stored form.
    "wav",
];

/// True when `extension` (no leading dot, already lowercased) is an allow-listed media extension
/// whose serve mime is an inert `image/`/`video/` type (never `text/html`, `image/svg+xml`, etc.).
pub(super) fn is_safe_upload_extension(extension: &str) -> bool {
    SAFE_UPLOAD_EXTENSIONS.contains(&extension)
}

/// Pick the stored extension for an upload **without trusting the client filename** (sc-8872).
///
/// The stored extension is what re-derives the serve mime later, so it must reflect the actual
/// media type, not whatever the client named the file. Precedence:
/// 1. The declared/sniffed `mime_type`'s canonical safe extension, when we recognize the mime.
/// 2. Otherwise the client filename's extension, but only if it is on the media allow-list.
/// 3. Otherwise `.bin` (serves `application/octet-stream` — inert), never an attacker-chosen
///    `.html`/`.svg`/`.js`.
///
/// A `video/mp4` upload named `evil.html` therefore stores `.mp4` (rule 1); a genuinely unknown
/// mime with a dangerous extension neutralizes to `.bin` (rule 3).
pub(super) fn upload_extension(filename: &str, mime_type: &str) -> String {
    if mime_type == crate::tensor_artifact::MIME {
        return ".npy".to_owned();
    }
    // 1. Prefer the extension the declared/sniffed mime canonically maps to, so the stored file's
    //    serve mime round-trips back to the type we actually accepted. mime_guess lists candidates;
    //    take the first one that is on our media allow-list.
    if let Some(extension) = mime_guess::get_mime_extensions_str(mime_type)
        .into_iter()
        .flatten()
        .map(|value| value.to_ascii_lowercase())
        .find(|value| is_safe_upload_extension(value))
    {
        return format!(".{extension}");
    }

    // 2. Fall back to the client extension only when it is itself an allow-listed media extension —
    //    never a document/script extension the client renamed the file to.
    if let Some(extension) = Path::new(filename)
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .filter(|value| is_safe_upload_extension(value))
    {
        return format!(".{extension}");
    }

    // 3. Anything else neutralizes to an inert, non-renderable extension.
    ".bin".to_owned()
}

impl ProjectStore {
    pub fn import_asset(&self, project_id: &str, upload: UploadAsset) -> ProjectStoreResult<Value> {
        if fs::metadata(&upload.source_path)?.len() == 0 {
            return Err(ProjectStoreError::BadRequest(
                "Uploaded file is empty".to_owned(),
            ));
        }
        // `extra` is an object in the sidecar contract, and `provenance` is written into it, so a
        // provenance that is not an object is refused here rather than reshaped or ignored.
        //
        // It used to be written through verbatim, which made `extra` a scalar — and once sc-15949
        // had a second writer of that slot, it became a silent-loss path: a `provenance` of `42`,
        // `"editor"`, `[1,2]` or `null` would take the whole slot and throw away a workflow the
        // reader had already found in the file. Rejecting is the only answer that neither loses the
        // envelope nor quietly rewrites what the caller sent. `apps/rust-api/src/assets.rs` refuses
        // the same shape at the multipart boundary, so this is the belt to that braces; the API is
        // the only production caller.
        if upload
            .provenance
            .as_ref()
            .is_some_and(|provenance| !provenance.is_object())
        {
            return Err(ProjectStoreError::BadRequest(
                "Upload provenance must be a JSON object".to_owned(),
            ));
        }
        let (project_path, _project_guard) = self.lock_project(project_id)?;
        let upload_dir = project_path.join("assets").join("uploads");
        fs::create_dir_all(&upload_dir)?;

        let guessed_mime = guess_mime_from_filename(&upload.filename);
        let content_type = upload
            .content_type
            .as_deref()
            .filter(|value| !value.is_empty() && *value != "application/octet-stream")
            .map(str::to_owned)
            .or(guessed_mime)
            .unwrap_or_else(|| "application/octet-stream".to_owned());
        let content_type = if Path::new(&upload.filename)
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("npy"))
        {
            crate::tensor_artifact::MIME.to_owned()
        } else {
            content_type
        };
        let tensor = if content_type == crate::tensor_artifact::MIME {
            Some(
                crate::tensor_artifact::inspect(&upload.source_path)
                    .map_err(ProjectStoreError::BadRequest)?,
            )
        } else {
            None
        };
        // SVG is active document syntax, not a generic upload format. Vector assets may enter
        // only through the worker-owned `vector_generate` sanitizer/rasterizer boundary, which
        // writes a canonical SVG plus a bounded PNG preview together (sc-22251).
        if content_type.eq_ignore_ascii_case("image/svg+xml")
            || Path::new(&upload.filename)
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("svg"))
        {
            return Err(ProjectStoreError::BadRequest(
                "SVG uploads are not supported; use Vector Studio".to_owned(),
            ));
        }
        if !content_type.starts_with("image/")
            && !content_type.starts_with("video/")
            && !content_type.starts_with("audio/")
            && content_type != crate::tensor_artifact::MIME
        {
            return Err(ProjectStoreError::BadRequest(
                "Only image, video and audio uploads are supported".to_owned(),
            ));
        }

        // sc-18650: audio is normalized by a DIFFERENT rule than image/video, so it is dispatched
        // here rather than folded into `normalize_image_upload` (which passes every non-image
        // through untouched). See `normalize_audio_upload` for why an uploaded clip is always
        // converted rather than stored as sent.
        //
        // sc-6143: transcode a valid-but-unsupported image (AVIF/HEIC/HEIF/TIFF/BMP/GIF) to lossless
        // PNG before storing, so every downstream decode site, thumbnail, and preview reads a format
        // it supports. Supported formats and videos pass through unchanged.
        let normalized = if content_type.starts_with("audio/") {
            normalize_audio_upload(&upload.source_path, &upload_dir)?
        } else {
            normalize_image_upload(
                &upload.source_path,
                &content_type,
                &upload.filename,
                &upload_dir,
                // sc-15949: a shared image may carry the sanitized recipe that made it. Read it from
                // the bytes the user handed us, before any transcode can strip it.
                WorkflowScan::Read,
            )?
        };
        let content_type = normalized.content_type.clone();

        let asset_id = format!("asset_{}", random_hex(16)?);
        let created_at = utc_now();
        let extension = normalized.extension.clone();
        let suffix = &asset_id[asset_id.len().saturating_sub(8)..];
        let filename = format!(
            "{}-{suffix}{extension}",
            safe_filename(&upload.filename, &asset_id)
        );
        let media_path = upload_dir.join(filename);
        let index_mutation = AssetIndexMutation::begin(&project_path)?;
        move_normalized_upload(&normalized, &upload.source_path, &media_path)?;
        let media_rel = relative_string(&project_path, &media_path)?;
        let display_name = Path::new(&upload.filename)
            .file_name()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| {
                media_path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("upload")
            })
            .to_owned();

        let media_type = media_type_for_mime(&content_type)?;
        let mut file = json!({
            "path": media_rel,
            "mimeType": content_type,
            "width": Value::Null,
            "height": Value::Null,
            "duration": Value::Null,
            "fps": Value::Null
        });
        // An audio upload's `file` block additionally carries the clip facts the audio tile, the
        // transport and the waveform all read — the SAME three fields `build_audio_sidecar_parts`
        // writes for an Audio Studio render, so the two doors an audio asset can come through
        // produce one shape (sc-18650). Without them the tile renders `0:00` and the waveform has
        // nothing to size against.
        //
        // MEASURED off the stored WAV, never trusted from the upload: the transcode above is what
        // decided them. They are added only for audio, so no image/video sidecar grows a key.
        if media_type == "audio" {
            if let Some(facts) = wav_facts(&media_path) {
                file["duration"] = json!(facts.duration_secs);
                file["sampleRate"] = json!(facts.sample_rate);
                file["channels"] = json!(facts.channels);
            }
        }
        let mut asset = json!({
            "schemaVersion": 1,
            "id": asset_id,
            "projectId": project_id,
            "generationSetId": Value::Null,
            "type": media_type,
            "displayName": display_name,
            "createdAt": created_at,
            // Manual imports are Library media in their own right (sc-2024).
            "origin": "upload",
            "file": file,
            "status": {
                "favorite": false,
                "rating": 0,
                "rejected": false,
                "trashed": false
            },
            "recipe": {
                "mode": "upload",
                "model": "manual-import",
                "adapter": "api-upload",
                "prompt": upload.filename,
                "negativePrompt": "",
                "seed": 0,
                "loras": [],
                "stylePreset": "none",
                "normalizedSettings": {},
                "rawAdapterSettings": { "contentType": content_type }
            },
            "lineage": {
                "parents": upload
                    .source_asset_id
                    .clone()
                    .map(|id| vec![Value::String(id)])
                    .unwrap_or_default(),
                "sourceAssetId": upload
                    .source_asset_id
                    .clone()
                    .map(Value::String)
                    .unwrap_or(Value::Null),
                "sourceTimestamp": Value::Null,
                "jobId": Value::Null
            }
        });
        if let Some(tensor) = tensor {
            asset["tensor"] = serde_json::to_value(&tensor)?;
            asset["file"]["contentHash"] = json!(format!("sha256:{}", tensor.sha256));
        }
        // Provenance (e.g. the Image Editor edit chain) rides the same top-level
        // `extra` slot generated assets use, so the Library/lineage UI can read it.
        //
        // sc-15949 shares that slot: an imported image that carried a workflow chunk records the
        // parsed, sanitized envelope under `extra.importedWorkflow`. The RECORD stays honest — the
        // stub recipe above is untouched and `recipe.mode` is still `"upload"`, because this image
        // WAS uploaded and was not generated here. sc-15952 offers the envelope as "Use this
        // recipe"; nothing here fabricates a `text_to_image` mode, a seed we did not run, or a
        // generation set that never existed.
        if let Some(object) = asset.as_object_mut() {
            // A non-object provenance was already refused at the top of this function, so this is
            // the only shape left.
            let provenance = upload.provenance.as_ref().and_then(Value::as_object);
            let mut extra = provenance.cloned().unwrap_or_default();
            // Cleared UNCONDITIONALLY, before the insert below and whether or not there is anything
            // to insert. `extra.importedWorkflow` is defined as the envelope THIS import read and
            // sanitized, so the key must be either ours or absent — never free JSON a caller put
            // under it in `provenance`.
            //
            // Inserting ours last is not enough on its own, and that was the bug: when the file
            // carries no chunk there is nothing to insert, and a caller-supplied
            // `provenance.importedWorkflow` was written through untouched — an unsanitized `model`,
            // an `advanced` map the allow-list would have emptied, and a `loras[].repo` of
            // `../../../../etc/passwd` in the one field sc-15950/sc-15952 turn into a cache path.
            // The guard has to be a removal, not an ordering.
            extra.remove(IMPORTED_WORKFLOW_KEY);
            // Upload/edit metadata is not authority for model identity or inherited obligations.
            extra.remove(crate::model_artifacts::provenance::MODEL_PROVENANCE_KEY);
            if let Some(workflow) = normalized
                .workflow
                .as_ref()
                .and_then(|share| serde_json::to_value(share).ok())
            {
                extra.insert(IMPORTED_WORKFLOW_KEY.to_owned(), workflow);
            }
            // An explicit `provenance` keeps its slot even when it reduces to nothing, because
            // `provenance: {}` wrote `extra: {}` before this story and "no behaviour change" means
            // the empty object too. With no provenance and no workflow, `extra` stays absent.
            if provenance.is_some() || !extra.is_empty() {
                object.insert("extra".to_owned(), Value::Object(extra));
            }
        }
        self.inherit_model_provenance(&project_path, &mut asset)?;
        let sidecar_path = media_path.with_extension("sceneworks.json");
        write_json(&sidecar_path, &asset)?;
        index_asset(project_id, &project_path, &asset, Some(&sidecar_path))?;
        index_mutation.commit();
        normalize_asset(project_id, &project_path, &sidecar_path)
    }
}
