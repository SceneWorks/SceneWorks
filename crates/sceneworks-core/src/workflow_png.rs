//! The PNG `iTXt` codec for the sanitized workflow envelope (sc-15947, epic 15945).
//!
//! [`crate::workflow_share`] owns the *contract*; this module owns the *plumbing*. It is the one
//! place in the tree that writes or reads a PNG text chunk, and it is deliberately standalone: the
//! worker call sites do not change until sc-15948, and the drag-to-load reader arrives in
//! sc-15952. Everything here is symmetric, so a round trip is testable without either.
//!
//! # Why a new dependency
//!
//! `image` 0.25's `PngEncoder` exposes no text-chunk API, and every generated image goes through
//! `save_with_format`, which emits IHDR/IDAT/IEND and nothing else. The repo writes no PNG text
//! chunks, no EXIF and no XMP anywhere today. `png` — already in the graph as `image`'s own PNG
//! codec — has both halves: `ITXtChunk` + [`png::Writer::write_text_chunk`] on the way out, and a
//! *bounded* text decompressor on the way in. See the root `Cargo.toml` for why it is declared at
//! the workspace level rather than here.
//!
//! # Why `iTXt`, compressed
//!
//! `tEXt` is Latin-1. A prompt is user prose and routinely is not: CJK, emoji, accented Latin. A
//! `tEXt` chunk would silently mangle all of it, which rules it out on its own —
//! `a_non_ascii_prompt_survives_the_round_trip` in `tests/workflow_png.rs` is the test that says
//! so. `iTXt` is UTF-8 and optionally deflate-compressed; compression matters because sc-15948
//! writes this into *every* generated image, and JSON of this shape compresses about 5×.
//!
//! # The read side is a trust boundary
//!
//! Writing is our own data. Reading is a file from a stranger — the whole point of the epic is
//! that these images travel. So:
//!
//! * [`MAX_WORKFLOW_TEXT_BYTES`] caps the DECOMPRESSED text, so what an attacker's ratio buys is
//!   our cap and not their claim: `ITXtChunk::decompress_text_with_limit` inflates in bounded steps
//!   and gives up, so a 1000:1 zip bomb costs a megabyte, not gigabytes.
//! * [`MAX_METADATA_BYTES`] bounds what the PNG decoder buffers out of the file's ancillary chunks
//!   by OUR budget rather than by the length a chunk declares, so a chunk header that *claims*
//!   2 GiB costs megabytes and is then refused.
//! * A PNG with no workflow chunk is `Ok(None)`, never an error. That is the common case for every
//!   image the user did not generate here, and sc-15952 must not shout about it.
//! * Everything else — truncated, corrupt, non-PNG, duplicate chunk, non-UTF-8, JSON of the wrong
//!   shape — is a typed [`WorkflowChunkError`] with a sentence fit to show a user.
//!
//! Parsing goes through [`parse_workflow_share_json`] and nothing else, so the sc-15946 allow-list
//! sanitizer runs on the way in. There is deliberately no second, laxer parse path; a structural
//! test in `tests/workflow_png.rs` fails the build if one appears.

use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Cursor, Seek, Write};
use std::path::Path;

use image::{ImageFormat, RgbImage};
use png::text_metadata::ITXtChunk;

use crate::workflow_share::{
    parse_workflow_share_json, WorkflowShare, WorkflowShareError, WORKFLOW_KIND_IMAGE,
};

/// The `iTXt` keyword the envelope is published under.
///
/// Namespaced on purpose. The two conventions already in the wild are Automatic1111's
/// `parameters` and ComfyUI's `prompt` / `workflow`, both unprefixed; a bare `workflow` would
/// collide with the second and make our chunk indistinguishable from theirs. 19 bytes, printable
/// ASCII, no spaces — inside the PNG spec's 1..=79 byte Latin-1 keyword rule with room to spare.
pub const WORKFLOW_CHUNK_KEYWORD: &str = "sceneworks:workflow";

/// Hard cap on the DECOMPRESSED chunk text, in bytes. 1 MiB.
///
/// Derived from the contract rather than picked round: [`crate::workflow_share`] already bounds
/// every string an envelope can carry, and its own worst case is what this has to clear. Six
/// prose fields (`prompt`, `negativePrompt`, `advanced.stylePrompt`, `advanced.systemMessage`, and
/// the structured prompt's `intent` / `runtimePrompt`) at `PROSE_MAX_BYTES` = 16 KiB each — the
/// bound is in BYTES, so 96 kB of prose is the ceiling in every script rather than a figure that
/// swings with the encoding — plus pose coordinate arrays and a few dozen bounded labels. Call it
/// ~150 kB for an envelope that is legitimate but absurd, and the serialized whole is refused past
/// `WORKFLOW_SHARE_MAX_BYTES` = 160 KiB regardless. 1 MiB clears that with room for the framing,
/// and is 2× *under* `png`'s own 2 MiB `DECOMPRESSION_LIMIT` default,
/// which is a generic guess where this is a bound on a known shape.
///
/// The cap is what makes a zip bomb cheap to refuse: `decompress_text_with_limit` inflates into a
/// buffer that grows in 32 KiB steps up to this ceiling and then stops, so the attacker's ratio
/// buys them 1 MiB of our memory no matter how large the payload claims to expand to.
pub const MAX_WORKFLOW_TEXT_BYTES: usize = 1024 * 1024;

/// Approximate ceiling on what the PNG decoder may allocate while buffering an untrusted file's
/// ancillary chunks. 8 MiB.
///
/// Distinct from [`MAX_WORKFLOW_TEXT_BYTES`] because it guards a different thing. A PNG chunk
/// header can declare up to 2 GiB; without a limit the decoder would buffer whatever the header
/// claims before anyone looked at the contents. `png` meters that against this budget and returns
/// `LimitsExceeded` instead, so the refusal happens in the megabytes.
///
/// Approximate, not exact, and in a direction worth being honest about: `png` grows the buffer and
/// only THEN finds it over budget, so the peak is reached rather than avoided, and the chunk's own
/// framing rides above the ceiling. A measured refusal of an over-budget `iTXt` chunk peaks at
/// 8,408,413 bytes live, including a single 8,388,709-byte allocation — `MAX_METADATA_BYTES` plus
/// 101 bytes of keyword and separator framing. What the budget buys is a bound that follows OUR
/// number instead of the attacker's (a chunk declaring 64 MiB peaks at the same ~8 MiB as one
/// declaring 16 MiB), not the absence of the allocation.
///
/// It is a *budget*, not a per-chunk size, and a CUMULATIVE one: `png`'s `Limits::reserve_bytes`
/// only ever subtracts, so every ancillary chunk in the file draws down the same pool, and one
/// chunk is charged more than once (each doubling of its growth buffer, then again when it parses).
/// The effective per-chunk ceiling is therefore a fraction of this number and depends on what came
/// before it. That imprecision is fine — the job is to be finite and far below what a header can
/// claim, not to be an exact quota. 8 MiB is ~8× the text cap (ample for a legitimate compressed
/// chunk, which runs to a couple of kilobytes) and 8× below `png`'s own 64 MiB default, so a
/// third-party PNG carrying fat EXIF still reads rather than erroring.
///
/// Being cumulative is what makes exhausting it in the post-IDAT tail an ABSENCE rather than an
/// error; see `workflow_chunk_after_image_data` for why the two passes treat it differently.
pub const MAX_METADATA_BYTES: usize = 8 * 1024 * 1024;

/// The 8-byte PNG signature. Checked before the decoder runs so "this is not a PNG" is its own
/// sentence rather than a generic decode failure.
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a workflow chunk could not be written or read.
///
/// Every variant is a sentence a user can act on, for the same reason [`WorkflowShareError`] is:
/// the read side is fed files from strangers, and a raw `png` decode message is not an answer to
/// "why won't this image load its recipe?".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowChunkError {
    /// Opening, creating or reading the file failed.
    Io { detail: String },
    /// The bytes do not start with the PNG signature (or are too short to).
    NotPng,
    /// The PNG itself is truncated or malformed — the chunk framing never got as far as our
    /// keyword.
    Png { detail: String },
    /// More than one chunk carries [`WORKFLOW_CHUNK_KEYWORD`].
    ///
    /// Refused rather than resolved. The PNG spec permits repeated `iTXt` keywords (they are meant
    /// to differ by language tag), but two workflows under our keyword is not a translation — it
    /// is a file someone appended a second recipe to. Picking one silently means showing a recipe
    /// that may not be the one that made the image, which is worse than saying the file is
    /// ambiguous.
    DuplicateChunk { count: usize },
    /// The file's metadata wanted more than [`MAX_METADATA_BYTES`] before our chunk was reached —
    /// the oversized-declared-length case.
    ///
    /// Raised by the PRIMARY pre-IDAT walk only. A budget already exhausted by the time the
    /// post-IDAT tail is walked is an absence instead, because the primary walk has by then already
    /// answered "no workflow"; `workflow_chunk_after_image_data` documents the asymmetry.
    MetadataTooLarge { limit: usize },
    /// An uncompressed chunk whose text is over the cap. Measurable, unlike the compressed case.
    TextTooLarge { bytes: usize, limit: usize },
    /// The chunk is present but its text could not be recovered.
    ///
    /// Deliberately coarse: this is the zip bomb (inflating past `limit`), a corrupt zlib stream,
    /// and bytes that inflate to something that is not UTF-8, all in one variant. `png` keeps its
    /// `TextDecodingError` crate-private, so the three are not distinguishable without matching on
    /// an upstream error *string* — a classification that would rot silently the next time that
    /// wording changed. `detail` carries the upstream message for the log; the cap is proven by
    /// `text_one_byte_under_the_cap_still_decompresses` / `a_zip_bomb_refuses_instead_of_inflating`
    /// straddling the threshold, not by reading this variant.
    UnreadableChunk { limit: usize, detail: String },
    /// The chunk decoded to text, but the text is not a workflow envelope this build can read.
    Envelope(WorkflowShareError),
    /// Encoding failed. Serializing the envelope, or `png` rejecting the image.
    Encode { detail: String },
}

impl fmt::Display for WorkflowChunkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { detail } => write!(f, "Could not read or write the image file ({detail})."),
            Self::NotPng => write!(f, "This file is not a PNG, so it carries no workflow."),
            Self::Png { detail } => write!(
                f,
                "This PNG could not be read far enough to look for a workflow ({detail})."
            ),
            Self::DuplicateChunk { count } => write!(
                f,
                "This PNG carries {count} `{WORKFLOW_CHUNK_KEYWORD}` chunks, so which workflow \
                 made the image is ambiguous."
            ),
            Self::MetadataTooLarge { limit } => write!(
                f,
                "This PNG's embedded metadata is larger than the {limit} bytes SceneWorks will \
                 read, so it was not searched for a workflow."
            ),
            Self::TextTooLarge { bytes, limit } => write!(
                f,
                "This PNG's workflow text is {bytes} bytes, over the {limit}-byte limit."
            ),
            Self::UnreadableChunk { limit, detail } => write!(
                f,
                "This PNG's workflow text could not be read as UTF-8 within {limit} bytes \
                 ({detail})."
            ),
            Self::Envelope(error) => write!(f, "{error}"),
            Self::Encode { detail } => {
                write!(f, "Could not write the workflow into the PNG ({detail}).")
            }
        }
    }
}

impl std::error::Error for WorkflowChunkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Envelope(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WorkflowShareError> for WorkflowChunkError {
    fn from(error: WorkflowShareError) -> Self {
        Self::Envelope(error)
    }
}

fn io_error(error: &std::io::Error) -> WorkflowChunkError {
    WorkflowChunkError::Io {
        detail: error.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

/// Encode `rgb` to `path` as a PNG, embedding `share` as a compressed `iTXt` chunk when there is
/// one.
///
/// `None` is not a degenerate case, it is the opt-out: it writes the image through the *same*
/// `save_with_format` call the worker uses today, so the bytes are identical to what SceneWorks
/// produces now. `the_none_path_is_byte_identical_to_save_with_format` in `tests/workflow_png.rs`
/// compares the two files byte for byte.
///
/// `Some` cannot call `save_with_format` — `image`'s `PngEncoder` has no text-chunk API at all —
/// so it goes through `png` directly. That makes it the arm that *could* change the encoder, and
/// the guarantee is deliberately just as strong: its output is the `None` output plus the inserted
/// `iTXt` chunk, byte for byte, with the encoder settings mirrored from `image` in
/// `encode_png_with_text` (not linked: it is private, and rustdoc warns on a public doc linking to
/// one). `the_some_path_is_the_none_path_plus_the_chunk` proves it by stripping the chunk back out
/// and comparing the remainder.
///
/// Both halves matter to sc-15948: opting out changes nothing about a user's files, and opting *in*
/// costs the chunk and nothing else — no encode-time or file-size change smuggled in alongside.
///
/// The chunk goes between IHDR and IDAT, where a reader finds it without decoding any pixels.
///
/// # Errors
/// [`WorkflowChunkError::Io`] if the file cannot be written, [`WorkflowChunkError::Encode`] if
/// serializing the envelope or encoding the PNG fails.
pub fn write_workflow_chunk(
    rgb: &RgbImage,
    path: &Path,
    share: Option<&WorkflowShare>,
) -> Result<(), WorkflowChunkError> {
    let Some(share) = share else {
        return rgb
            .save_with_format(path, ImageFormat::Png)
            .map_err(|error| WorkflowChunkError::Io {
                detail: error.to_string(),
            });
    };
    let text = serde_json::to_string(share).map_err(|error| WorkflowChunkError::Encode {
        detail: error.to_string(),
    })?;
    let file = File::create(path).map_err(|error| io_error(&error))?;
    let mut sink = BufWriter::new(file);
    encode_png_with_text(rgb, &mut sink, &[workflow_chunk(text)])?;
    sink.flush().map_err(|error| io_error(&error))?;
    Ok(())
}

/// The compressed `iTXt` chunk for one envelope.
///
/// `ITXtChunk::new` defaults to `compressed: false`; flipping it is what makes the encoder deflate
/// the text on the way out. That is the whole reason `png::Encoder::add_itxt_chunk` is not used —
/// it writes the chunk uncompressed, and this rides in every generated image.
fn workflow_chunk(text: String) -> ITXtChunk {
    let mut chunk = ITXtChunk::new(WORKFLOW_CHUNK_KEYWORD, text);
    chunk.compressed = true;
    chunk
}

/// Encode an 8-bit RGB PNG with `chunks` written between IHDR and IDAT.
///
/// Split out from [`write_workflow_chunk`] so the adversarial tests can build a PNG carrying any
/// text chunk they like — a wrong keyword, two of ours, an uncompressed one, a payload that
/// inflates past the cap — through the same encoder that writes the real thing.
///
/// # The settings are a mirror of `image`'s, on purpose
///
/// This is the one place the two write paths can drift, and drifting is not a cosmetic risk:
/// sc-15948 turns this arm on for *every* generated image, so anything the settings change here
/// becomes a change to every file SceneWorks produces. Adding a text chunk must cost a text chunk
/// and nothing else — not encode time, not output size.
///
/// `png::Encoder::new` starts from `Options::default()`, whose compression is
/// `DeflateCompression::default()` = `from_simple(Compression::Balanced)` = flate2 level 6
/// (`png-0.18.1/src/common.rs`). `image` does not use that default: its `PngEncoder::new` takes
/// `CompressionType::default()`, and `CompressionType::Fast` is the variant carrying `#[default]`
/// (`image-0.25.10/src/codecs/png.rs`), which `encode_inner` maps to `png::Compression::Fast` →
/// `DeflateCompression::FdeflateUltraFast`. Left unset, `Some` would silently encode ~8× slower and,
/// on the kind of image SceneWorks actually renders, produce a file a little over half the size
/// (measured at 1024²: 92,526 bytes at `Balanced` against 167,360 at `Fast`, a 1.81× reduction) — a
/// product change nobody asked this story for.
///
/// So both calls below mirror `image`'s `encode_inner` exactly, in its order:
///
/// * `Compression::Fast` for `CompressionType::Fast`.
/// * `Filter::Adaptive` for `FilterType::Adaptive`, the other `#[default]`.
///
/// The filter call looks redundant — `set_compression` already applies
/// `Filter::from_simple(Fast)`, which is `Adaptive` today — and it is kept anyway because `image`
/// pins the filter explicitly *after* setting compression. If `png` ever remapped
/// `from_simple(Fast)`, `image` would keep writing `Adaptive` and this would follow it rather than
/// diverge. Mirroring the call sequence is what makes that true; mirroring only the outcome would
/// not. The filter is not a cosmetic setting either: on a compressible render at 1024², `Adaptive`
/// writes 167,360 bytes where `Sub` writes 388,103 and `NoFilter` 3,147,060 — 2.3× and 18.8×.
///
/// None of this is trusted to stay true by inspection:
/// `the_some_path_is_the_none_path_plus_the_chunk` in `tests/workflow_png.rs` strips the `iTXt`
/// chunk back out of this encoder's output at the byte level and compares what is left against a
/// real `save_with_format` write of the same pixels. It runs over TWO fixtures, because the two
/// deflate regimes see different settings: an incompressible one, where `png` falls back to storing
/// rows and the filter is not consulted at all, and a smooth one that actually deflates, which is
/// the only regime in which a filter divergence is visible.
fn encode_png_with_text<W: Write>(
    rgb: &RgbImage,
    sink: W,
    chunks: &[ITXtChunk],
) -> Result<(), WorkflowChunkError> {
    let encode_error = |error: png::EncodingError| WorkflowChunkError::Encode {
        detail: error.to_string(),
    };
    let mut encoder = png::Encoder::new(sink, rgb.width(), rgb.height());
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    encoder.set_filter(png::Filter::Adaptive);
    let mut writer = encoder.write_header().map_err(encode_error)?;
    for chunk in chunks {
        writer.write_text_chunk(chunk).map_err(encode_error)?;
    }
    writer
        .write_image_data(rgb.as_raw())
        .map_err(encode_error)?;
    writer.finish().map_err(encode_error)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Strip
// ---------------------------------------------------------------------------

/// The three PNG text-chunk types a keyword can appear under.
///
/// [`write_workflow_chunk`] only ever emits `iTXt`, and [`read_workflow_chunk`] only ever reads
/// `iTXt` (`png` files a `tEXt` under `uncompressed_latin1_text` and a `zTXt` under
/// `compressed_latin1_text`, neither of which the reader looks at). The stripper is broader than
/// both on purpose: it is a privacy operation, and "we would not have read it back" is not a reason
/// to hand a stranger a chunk with our keyword on it. Being broader can only ever remove more.
const TEXT_CHUNK_TYPES: [&[u8; 4]; 3] = [b"iTXt", b"tEXt", b"zTXt"];

/// Bytes of PNG chunk framing that are not the payload: the 4-byte length, the 4-byte type and the
/// 4-byte CRC.
const CHUNK_OVERHEAD: usize = 12;

/// Excise every [`WORKFLOW_CHUNK_KEYWORD`] text chunk from PNG `bytes`, without re-encoding
/// anything (sc-15953).
///
/// `Ok(None)` means there was nothing to take out and the caller should use the source bytes
/// unchanged — a PNG with no workflow chunk, or a file that is not a PNG at all. The second case is
/// not a failure: [`write_workflow_chunk`] only ever writes into a PNG, so a JPEG or an MP4 cannot
/// be carrying one, and a "save without the workflow" of a video has nothing to do.
///
/// # Why excise rather than decode and re-encode
///
/// A re-encode is a *new* file that happens to have the same pixels. This is the old file minus a
/// slice: the signature, IHDR, PLTE, every IDAT and IEND are copied through byte for byte, so the
/// pixels are not merely equal — the compressed image data is the same bytes it was, and no encoder
/// setting, filter choice or `image`/`png` version is in the path at all. Each PNG chunk carries its
/// own CRC over its own type and data, so removing one whole chunk leaves every other chunk's CRC
/// still correct; nothing is recomputed.
///
/// `strip_of_an_embedded_png_is_byte_identical_to_the_none_path` in `tests/workflow_png.rs` proves
/// it the strongest way available: stripping the `Some` output yields the `None` output *byte for
/// byte*, over both DEFLATE regimes and four sizes.
///
/// # Errors
/// [`WorkflowChunkError::Png`] whenever the file cannot be accounted for chunk by chunk — see
/// [`workflow_chunk_spans`] for the three shapes that means. That is deliberately an error rather
/// than a quiet "nothing to strip": for a file we cannot walk, "there is no workflow chunk in it"
/// is not something we know, and a caller asking for a stripped copy must refuse rather than hand
/// over the original.
pub fn strip_workflow_chunk(bytes: &[u8]) -> Result<Option<Vec<u8>>, WorkflowChunkError> {
    let spans = workflow_chunk_spans(bytes)?;
    if spans.is_empty() {
        // No output buffer is allocated on this path at all. The caller uses the source bytes, and
        // the common case — a PNG with nothing of ours in it, or a file that is not a PNG — costs
        // the walk and nothing else.
        return Ok(None);
    }
    let mut out = Vec::with_capacity(bytes.len() - spans.iter().map(Span::len).sum::<usize>());
    for kept in kept_spans(bytes.len(), &spans) {
        out.extend_from_slice(&bytes[kept.start..kept.end]);
    }
    Ok(Some(out))
}

/// A half-open byte range of a PNG. A `std::ops::Range` in all but name; a named struct because
/// `Range` is not `Copy` and these are passed around by value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// First byte of the span.
    pub start: usize,
    /// One past the last byte.
    pub end: usize,
}

impl Span {
    /// How many bytes the span covers.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the span covers no bytes. Never true for a span this module produces.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// The byte spans a strip would remove from `bytes`, ascending and non-overlapping.
///
/// The ONE walk. [`strip_workflow_chunk`] builds its output from this, and so does the API's
/// download route — which serves the kept spans as slices of the buffer it already read rather
/// than copying them into a second one, so a strip costs the file once instead of twice. Two
/// callers, one set of rules; a hole closed here is closed in both.
///
/// An empty result means "there is nothing of ours in this file and we are sure of that", which is
/// what licenses a caller to use the source bytes unchanged.
///
/// # What it refuses, and why each one is a leak if it does not
///
/// Every byte of the file has to be accounted for. A chunk we walked is classified — ours and
/// removed, or someone else's and kept. Bytes we could NOT walk are the dangerous ones, because
/// "we did not find a workflow chunk there" and "there is no workflow chunk there" are different
/// statements and only the second one is safe to act on.
///
/// * **The walk never reaches IEND.** A PNG truncated at a chunk boundary used to walk clean:
///   every chunk parsed, our chunk was excised, and the result was an `Ok(Some(…))` PNG with no
///   IEND in it. "Save a copy without the workflow" then wrote an undecodable file and reported
///   success. There is no partial-credit answer here — refuse.
/// * **A tail after IEND that is not chunk framing and carries our keyword.** Trailing bytes past
///   IEND are not chunks under the spec, and a file with a tail we did not write should survive
///   the round trip, so a tail that cannot be walked is copied through. But copying through is
///   only honest when there is nothing of ours in it: a `sceneworks:workflow` chunk hidden in an
///   unwalkable tail was returned as `Ok(None)` — "use the source unchanged" — with the prompt
///   still in the file, and, when a strippable chunk sat pre-IDAT as well, as `Ok(Some(…))` and a
///   cheerful "I removed one". Our own reader stops at IEND so it would not read the leftover
///   back, which is exactly why this had to be found by grepping the bytes rather than by asking
///   the reader: a stranger's tooling and a raw byte search both find it, and that is the threat
///   model. So the unwalkable remainder is scanned for [`WORKFLOW_CHUNK_KEYWORD`] and refused if
///   it is there.
/// * **Framing that runs past the end of the file before IEND.** Unchanged: a chunk header
///   claiming more bytes than the file holds means the PNG is malformed and no copy can be vouched
///   for.
///
/// The keyword scan is deliberately confined to the unwalkable remainder rather than run over the
/// whole file. Every walked chunk was already classified on purpose, and a foreign chunk whose
/// *text* merely mentions our keyword — a note, a ComfyUI graph naming it — is a file we must
/// still be able to save. Only unaccounted-for bytes get the blunt instrument.
///
/// # Errors
/// [`WorkflowChunkError::Png`], with a detail naming which of the three it was.
pub fn workflow_chunk_spans(bytes: &[u8]) -> Result<Vec<Span>, WorkflowChunkError> {
    if bytes.len() < PNG_SIGNATURE.len() || bytes[..PNG_SIGNATURE.len()] != PNG_SIGNATURE {
        return Ok(Vec::new());
    }
    let mut spans: Vec<Span> = Vec::new();
    let mut cursor = PNG_SIGNATURE.len();
    let mut past_iend = false;

    while cursor < bytes.len() {
        let Some(end) = chunk_end(bytes, cursor) else {
            if !past_iend {
                return Err(WorkflowChunkError::Png {
                    detail: format!("the chunk at byte {cursor} runs past the end of the file"),
                });
            }
            let tail = &bytes[cursor..];
            if contains_workflow_keyword(tail) {
                return Err(WorkflowChunkError::Png {
                    detail: format!(
                        "{} bytes of trailing data at byte {cursor} are not chunk framing and \
                         carry `{WORKFLOW_CHUNK_KEYWORD}`, so a copy of this file cannot be \
                         vouched for",
                        tail.len()
                    ),
                });
            }
            // Benign trailing bytes. Not walkable, nothing of ours in them, so they ride through.
            return Ok(spans);
        };
        let kind = &bytes[cursor + 4..cursor + 8];
        if kind == b"IEND" {
            past_iend = true;
        }
        if is_workflow_text_chunk(kind, &bytes[cursor + 8..end - 4]) {
            spans.push(Span { start: cursor, end });
        }
        cursor = end;
    }

    if !past_iend {
        return Err(WorkflowChunkError::Png {
            detail: format!(
                "the chunk framing ends at byte {cursor} without an IEND, so this PNG is \
                 truncated and a copy of it would not decode"
            ),
        });
    }
    Ok(spans)
}

/// The complement of `removed` over `0..total`: the spans a strip keeps, ascending.
///
/// Shared with the API's download route so the served body and [`strip_workflow_chunk`]'s buffer
/// are assembled from the same arithmetic rather than two hand-rolled loops.
#[must_use]
pub fn kept_spans(total: usize, removed: &[Span]) -> Vec<Span> {
    let mut kept = Vec::with_capacity(removed.len() + 1);
    let mut cursor = 0_usize;
    for span in removed {
        if span.start > cursor {
            kept.push(Span {
                start: cursor,
                end: span.start,
            });
        }
        cursor = span.end;
    }
    if cursor < total {
        kept.push(Span {
            start: cursor,
            end: total,
        });
    }
    kept
}

/// Whether [`WORKFLOW_CHUNK_KEYWORD`] appears anywhere in `haystack`, as raw bytes.
///
/// The same thing a `grep` for our keyword would find, which is the point: the guard it backs is
/// about what a stranger's tooling can dig out of bytes we could not parse, not about what our own
/// reader would load.
fn contains_workflow_keyword(haystack: &[u8]) -> bool {
    let needle = WORKFLOW_CHUNK_KEYWORD.as_bytes();
    haystack.len() >= needle.len()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

/// One past the CRC of the chunk starting at `cursor`, or `None` when the header or the payload it
/// declares runs past the end of `bytes`.
fn chunk_end(bytes: &[u8], cursor: usize) -> Option<usize> {
    let header = bytes.get(cursor..cursor + 8)?;
    let length = usize::try_from(u32::from_be_bytes(header[..4].try_into().ok()?)).ok()?;
    let end = cursor.checked_add(CHUNK_OVERHEAD)?.checked_add(length)?;
    (end <= bytes.len()).then_some(end)
}

/// Whether a chunk of type `kind` carrying `data` is a text chunk under our keyword.
///
/// All three text types lay their payload out as `keyword NUL …`, so the keyword is the payload up
/// to its first NUL in every case. Matched byte-exactly: PNG keywords are case-sensitive, and
/// [`read_workflow_chunk`] resolves the keyword the same way, so `Sceneworks:Workflow` is a
/// different keyword to both halves.
fn is_workflow_text_chunk(kind: &[u8], data: &[u8]) -> bool {
    if !TEXT_CHUNK_TYPES.iter().any(|text| *text == kind) {
        return false;
    }
    let keyword = data.split(|byte| *byte == 0).next().unwrap_or_default();
    keyword == WORKFLOW_CHUNK_KEYWORD.as_bytes()
}

/// Copy `source` to `destination` with any [`WORKFLOW_CHUNK_KEYWORD`] chunk excised, and report
/// whether one was there (sc-15953).
///
/// The "Save As without the workflow" seam. A file with nothing to strip is copied through
/// [`std::fs::copy`] — the same call the plain Save As makes — so the two paths produce the same
/// file for a source that never carried a chunk, rather than one of them rewriting it.
///
/// # Errors
/// [`WorkflowChunkError::Io`] if the source cannot be read or the destination cannot be written,
/// and whatever [`strip_workflow_chunk`] returns for a PNG whose framing cannot be walked.
pub fn copy_without_workflow_chunk(
    source: &Path,
    destination: &Path,
) -> Result<bool, WorkflowChunkError> {
    let bytes = std::fs::read(source).map_err(|error| io_error(&error))?;
    match strip_workflow_chunk(&bytes)? {
        Some(stripped) => {
            std::fs::write(destination, stripped).map_err(|error| io_error(&error))?;
            Ok(true)
        }
        None => {
            std::fs::copy(source, destination).map_err(|error| io_error(&error))?;
            Ok(false)
        }
    }
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

/// Read the workflow envelope out of PNG bytes.
///
/// `Ok(None)` means "this is a readable PNG that carries no SceneWorks workflow" — the common
/// case for every image the user did not generate here, and not a failure.
///
/// The metadata BEFORE the first IDAT is searched first, because that is where
/// [`write_workflow_chunk`] puts ours: every SceneWorks image answers in one chunk walk that stops
/// at the image data. Only when that finds nothing is the tail after IDAT searched too — an `iTXt`
/// there is legal under the PNG spec, and sc-15949 reads files written by tools that are not ours,
/// so "we never write it there" is not a reason to be blind to it. Silently reporting no recipe for
/// a file that carries one is the worse failure.
///
/// Neither pass decodes a pixel. The second walks the remaining chunk framing with no output buffer,
/// which `png` handles by skipping the IDAT payload rather than inflating it, so the hardening
/// property survives: a hostile file still cannot make us decompress an image to look for text.
/// Both passes run under the same [`MAX_METADATA_BYTES`] budget, resolve the same keyword, and hand
/// whatever they find to the same capped decompress and the same single parser — the tail is not a
/// laxer route in, only another place to look. That budget is cumulative, so a file fat enough to
/// spend it before the tail is reached reads as `Ok(None)` rather than as an error; the tail is an
/// extra place to look, and failing to look there cannot turn an absence into a failure.
///
/// # Errors
/// See [`WorkflowChunkError`]. Never panics, never inflates more than
/// [`MAX_WORKFLOW_TEXT_BYTES`], and buffers metadata bounded at approximately
/// [`MAX_METADATA_BYTES`] — the budget plus the chunk framing that rides above it, rather than
/// whatever length a chunk header declares.
pub fn read_workflow_chunk(bytes: &[u8]) -> Result<Option<WorkflowShare>, WorkflowChunkError> {
    read_workflow_chunk_from(Cursor::new(bytes))
}

/// Read the workflow envelope out of a PNG file, streaming rather than slurping it.
///
/// # Errors
/// See [`read_workflow_chunk`].
pub fn read_workflow_chunk_file(path: &Path) -> Result<Option<WorkflowShare>, WorkflowChunkError> {
    let file = File::open(path).map_err(|error| io_error(&error))?;
    read_workflow_chunk_from(BufReader::new(file))
}

/// The one read implementation both public entry points share.
fn read_workflow_chunk_from<R: BufRead + Seek>(
    mut source: R,
) -> Result<Option<WorkflowShare>, WorkflowChunkError> {
    // Signature first. `png` would report a missing signature too, but as one more decode failure
    // among many; "this file is not a PNG" is a different thing to tell a user than "this PNG is
    // corrupt", and it costs 8 bytes to know which.
    let mut signature = [0_u8; PNG_SIGNATURE.len()];
    match source.read_exact(&mut signature) {
        Ok(()) => {}
        // Short of a signature is not a PNG. Anything else is a real I/O fault.
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(WorkflowChunkError::NotPng)
        }
        Err(error) => return Err(io_error(&error)),
    }
    if signature != PNG_SIGNATURE {
        return Err(WorkflowChunkError::NotPng);
    }
    source.rewind().map_err(|error| io_error(&error))?;

    let mut decoder = png::Decoder::new_with_limits(
        source,
        png::Limits {
            bytes: MAX_METADATA_BYTES,
        },
    );
    // An embedded colour profile is the other unbounded ancillary chunk, and it is metadata we
    // never read. Skipping it keeps a legitimately fat ICC profile from spending the budget that
    // exists to bound our own chunk.
    decoder.set_ignore_iccp_chunk(true);
    // `read_info` stops at the first IDAT: all the metadata, none of the pixels.
    let mut reader = decoder.read_info().map_err(|error| match error {
        png::DecodingError::LimitsExceeded => WorkflowChunkError::MetadataTooLarge {
            limit: MAX_METADATA_BYTES,
        },
        error => WorkflowChunkError::Png {
            detail: error.to_string(),
        },
    })?;

    // Pass one: the pre-IDAT metadata, where our own writer puts the chunk. Every image SceneWorks
    // generated answers here and never touches the tail.
    let mut found = select_workflow_chunk(reader.info())?;
    // Pass two: the tail, and ONLY when pass one came up empty. That ordering is the precedence
    // rule as much as an optimization — a pre-IDAT chunk is the one our encoder wrote in the same
    // pass that wrote the pixels, so if both places carry one, the pre-IDAT chunk is the recipe
    // that made the image and a post-IDAT chunk is something appended afterwards.
    if found.is_none() {
        found = workflow_chunk_after_image_data(&mut reader)?;
    }
    let Some(mut chunk) = found else {
        return Ok(None);
    };

    chunk
        .decompress_text_with_limit(MAX_WORKFLOW_TEXT_BYTES)
        .map_err(|error| WorkflowChunkError::UnreadableChunk {
            limit: MAX_WORKFLOW_TEXT_BYTES,
            detail: error.to_string(),
        })?;
    // Now a clone of an already-inflated `String`. Calling `get_text()` on a chunk that is still
    // compressed would inflate it UNBOUNDED — the decompress above is what makes this line safe,
    // and swapping the two would reintroduce exactly the hole this module exists to close.
    let text = chunk
        .get_text()
        .map_err(|error| WorkflowChunkError::UnreadableChunk {
            limit: MAX_WORKFLOW_TEXT_BYTES,
            detail: error.to_string(),
        })?;
    // A chunk written with the compression flag CLEAR never went through the bounded inflate, so
    // it is bounded only by the metadata budget. This makes the two paths agree on one cap.
    if text.len() > MAX_WORKFLOW_TEXT_BYTES {
        return Err(WorkflowChunkError::TextTooLarge {
            bytes: text.len(),
            limit: MAX_WORKFLOW_TEXT_BYTES,
        });
    }

    // The ONLY parse path, so the sc-15946 allow-list sanitizer runs on every envelope that
    // arrives from outside. A second `serde_json::from_str::<WorkflowShare>` here would read the
    // same JSON without the value-level guards.
    let share = parse_workflow_share_json(&text)?;
    // A PNG carries an IMAGE workflow. Nothing else (sc-15956).
    //
    // `parse_workflow_share` validates the ENVELOPE and, since the video lane shipped, accepts
    // either kind — which is right for the contract and wrong for a consumer. Without this the
    // widened gate would have quietly changed THIS module's behaviour: a PNG carrying a video
    // envelope used to be refused, and would have started parsing, so every reader downstream
    // (`/workflows/inspect`, the Image Editor's header, the drag-to-load import) would have offered
    // a video recipe as an image prefill.
    //
    // So the container asserts its own kind, and `workflow_mp4::read_workflow_metadata` does the
    // symmetrical thing. Our writers only ever pair a PNG with an image envelope and an MP4 with a
    // video one, so a mismatch is a hand-made file or a future build's, and refusing is what the
    // `UnsupportedKind` sentence already says to a user.
    if share.kind != WORKFLOW_KIND_IMAGE {
        return Err(WorkflowChunkError::Envelope(
            WorkflowShareError::UnsupportedKind {
                found: share.kind,
                supported: WORKFLOW_KIND_IMAGE.to_owned(),
            },
        ));
    }
    Ok(Some(share))
}

/// Pick our chunk out of the `iTXt` chunks the decoder has parsed so far.
///
/// Shared by both passes so the keyword match and the duplicate rule cannot drift apart. The
/// returned chunk is CLONED because decompression mutates it in place and `Info` hands out a shared
/// reference; the clone is of the still-compressed bytes, which [`MAX_METADATA_BYTES`] already
/// bounded.
fn select_workflow_chunk(info: &png::Info<'_>) -> Result<Option<ITXtChunk>, WorkflowChunkError> {
    let matching: Vec<&ITXtChunk> = info
        .utf8_text
        .iter()
        .filter(|chunk| chunk.keyword == WORKFLOW_CHUNK_KEYWORD)
        .collect();
    match matching.as_slice() {
        [] => Ok(None),
        [only] => Ok(Some((*only).clone())),
        many => Err(WorkflowChunkError::DuplicateChunk { count: many.len() }),
    }
}

/// Pass two: walk the chunks after the image data and look for the keyword there.
///
/// `Reader::finish` reads on to IEND with no output buffer, and `png` responds to the absent buffer
/// by consuming the IDAT payload without handing it to the inflater
/// (`png-0.18.1/src/decoder/stream.rs`, the `ImageData` state). So this costs chunk framing and CRC
/// over bytes already on their way past — never an image decode, and never an allocation that
/// scales with the pixels. Ancillary chunks it passes are parsed under the limits the decoder was
/// built with, so the metadata budget still applies out here.
///
/// A walk that FAILS is reported as an absence rather than an error, with NO exception. Pass one
/// already succeeded and already told us there is no workflow in the metadata proper; a file whose
/// tail is truncated or corrupt does not change that answer, and turning "this PNG is a bit damaged
/// and has no recipe" into a hard error would break the `Ok(None)` promise for a case that reads
/// cleanly today. Discarding the error can only ever yield *fewer* workflows, never one that
/// bypassed a check, so this is not a soft edge on the trust boundary.
///
/// # Why an exhausted budget is an absence out here and an error in pass one
///
/// The asymmetry is deliberate. [`MAX_METADATA_BYTES`] is a CUMULATIVE budget — `png`'s
/// `Limits::reserve_bytes` only subtracts — so a file with several megabytes of legitimate but fat
/// pre-IDAT `iTXt` can pass through `read_info` and then leave too little of the pool to frame one
/// more chunk out here. Surfacing that as [`WorkflowChunkError::MetadataTooLarge`] would take a
/// class of foreign file that read as `Ok(None)` before this pass existed and turn it into a
/// user-visible error, which is exactly what the module contract forbids and what sc-15949 would hit
/// on import. The rendered sentence would also be wrong: pass one *did* search the file.
///
/// So the rule follows what we were asked for. A chunk too large in the PRIMARY walk is an error —
/// we were asked to read the metadata and could not. A budget exhausted by the time the TAIL is
/// reached is an absence — the primary walk already answered "no workflow", and the tail is an extra
/// place to look, not a second question. `MAX_METADATA_BYTES` still bounds what is buffered either
/// way, which is the property that matters: the allocation is sized by our budget rather than by the
/// length the chunk declares — a 64 MiB claim costs ~8 MiB — and only then is it refused.
fn workflow_chunk_after_image_data<R: BufRead + Seek>(
    reader: &mut png::Reader<R>,
) -> Result<Option<ITXtChunk>, WorkflowChunkError> {
    if reader.finish().is_err() {
        return Ok(None);
    }
    select_workflow_chunk(reader.info())
}

/// Size of the compressed `iTXt` chunk `share` would occupy in a PNG, in bytes — the whole chunk
/// as it appears on disk, including its 4-byte length, 4-byte type, keyword, separators and CRC.
///
/// Exists so the per-image cost of sc-15948 is a measured number rather than an estimate; it is
/// what `the_chunk_stays_a_text_chunk_and_not_a_payload` in `tests/workflow_png.rs` reports and
/// bounds.
///
/// # Errors
/// [`WorkflowChunkError::Encode`] if the envelope cannot be serialized or compressed.
pub fn workflow_chunk_size(share: &WorkflowShare) -> Result<usize, WorkflowChunkError> {
    use png::text_metadata::EncodableTextChunk;

    let text = serde_json::to_string(share).map_err(|error| WorkflowChunkError::Encode {
        detail: error.to_string(),
    })?;
    let mut encoded = Vec::new();
    workflow_chunk(text)
        .encode(&mut encoded)
        .map_err(|error| WorkflowChunkError::Encode {
            detail: error.to_string(),
        })?;
    Ok(encoded.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------
    // Fixtures
    // ---------------------------------------------------------------------

    /// A small deterministic image. Content is irrelevant to every test here; only that it encodes
    /// and that the pixels come back matters.
    fn rgb_fixture() -> RgbImage {
        RgbImage::from_fn(9, 7, |x, y| {
            image::Rgb([(x * 20) as u8, (y * 30) as u8, ((x + y) * 10) as u8])
        })
    }

    /// A PNG carrying exactly `chunks`, in memory.
    fn png_with(chunks: &[ITXtChunk]) -> Vec<u8> {
        let mut bytes = Vec::new();
        encode_png_with_text(&rgb_fixture(), &mut bytes, chunks).expect("fixture PNG encodes");
        bytes
    }

    /// A PNG with our keyword carrying `text`, compressed as the real writer does.
    fn png_with_workflow_text(text: &str) -> Vec<u8> {
        png_with(&[workflow_chunk(text.to_owned())])
    }

    /// The smallest envelope `parse_workflow_share_json` accepts, as JSON text.
    fn minimal_envelope_json(prompt: &str) -> String {
        serde_json::json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": { "name": "SceneWorks", "url": "https://example.invalid", "version": "0.8.1" },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": prompt,
        })
        .to_string()
    }

    // ---------------------------------------------------------------------
    // Hand-built chunk framing, for the cases the encoder cannot express
    // ---------------------------------------------------------------------
    //
    // `ITXtChunk` holds its text as a `String`, so it can never produce a chunk whose payload is
    // not valid UTF-8, nor one whose declared length lies about its contents. Those are exactly
    // two of the adversarial inputs, so the corpus below frames a few chunks by hand. Bitwise
    // CRC-32 and Adler-32 rather than a dependency: this is test scaffolding, and both are a
    // dozen lines.

    /// Byte offset one past the IHDR chunk: signature + (length + type + 13 bytes + CRC).
    const AFTER_IHDR: usize = 8 + 4 + 4 + 13 + 4;

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFF_u32;
        for byte in bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                let mask = if crc & 1 == 1 { 0xEDB8_8320 } else { 0 };
                crc = (crc >> 1) ^ mask;
            }
        }
        !crc
    }

    fn adler32(bytes: &[u8]) -> u32 {
        let (mut low, mut high) = (1_u32, 0_u32);
        for byte in bytes {
            low = (low + u32::from(*byte)) % 65_521;
            high = (high + low) % 65_521;
        }
        (high << 16) | low
    }

    /// A framed PNG chunk. `declared_len` overrides the length field so a chunk can lie about how
    /// much data follows it.
    fn framed_chunk(kind: &[u8; 4], data: &[u8], declared_len: Option<u32>) -> Vec<u8> {
        let length = declared_len.unwrap_or_else(|| u32::try_from(data.len()).expect("fits u32"));
        let mut checked = kind.to_vec();
        checked.extend_from_slice(data);
        let mut out = length.to_be_bytes().to_vec();
        out.extend_from_slice(&checked);
        out.extend_from_slice(&crc32(&checked).to_be_bytes());
        out
    }

    /// The `iTXt` data layout: keyword NUL flag method language NUL translated NUL text.
    fn itxt_data(keyword: &str, compressed: bool, text: &[u8]) -> Vec<u8> {
        let mut data = keyword.as_bytes().to_vec();
        data.push(0); // keyword separator
        data.push(u8::from(compressed)); // compression flag
        data.push(0); // compression method: zlib
        data.push(0); // empty language tag + separator
        data.push(0); // empty translated keyword + separator
        data.extend_from_slice(text);
        data
    }

    /// A zlib stream of one STORED (uncompressed) deflate block. The only way to hand a bounded
    /// inflater bytes that are not valid UTF-8 without a compressor in the dependency graph.
    fn zlib_stored(payload: &[u8]) -> Vec<u8> {
        let length = u16::try_from(payload.len()).expect("stored block fits u16");
        // CMF 0x78 / FLG 0x01: deflate, 32 KiB window, no preset dictionary, and (0x78 << 8 | 1)
        // is divisible by 31 as the zlib header check requires.
        let mut out = vec![0x78, 0x01, 0x01];
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(&(!length).to_le_bytes());
        out.extend_from_slice(payload);
        out.extend_from_slice(&adler32(payload).to_be_bytes());
        out
    }

    /// Splice a hand-framed chunk in right after IHDR, where the writer puts ours.
    fn splice_after_ihdr(png: &[u8], chunk: &[u8]) -> Vec<u8> {
        let mut out = png[..AFTER_IHDR].to_vec();
        out.extend_from_slice(chunk);
        out.extend_from_slice(&png[AFTER_IHDR..]);
        out
    }

    /// Byte offset of the IEND chunk header, found by walking the framing rather than assuming it
    /// is the last twelve bytes.
    fn iend_offset(png: &[u8]) -> usize {
        let mut cursor = 8;
        loop {
            assert!(
                cursor + 8 <= png.len(),
                "walked off the end looking for IEND"
            );
            if &png[cursor + 4..cursor + 8] == b"IEND" {
                return cursor;
            }
            let length = usize::try_from(u32::from_be_bytes(
                png[cursor..cursor + 4].try_into().expect("a length field"),
            ))
            .expect("a chunk length fits usize");
            cursor += 12 + length;
        }
    }

    /// Splice a hand-framed chunk in AFTER the image data, between the last IDAT and IEND.
    ///
    /// A legal home for an ancillary chunk under the PNG spec, and the one `read_info` alone cannot
    /// see, because it stops at the first IDAT. Our own writer never puts a chunk here — these
    /// fixtures stand in for the foreign writers sc-15949 will read.
    fn splice_before_iend(png: &[u8], chunk: &[u8]) -> Vec<u8> {
        let iend = iend_offset(png);
        let mut out = png[..iend].to_vec();
        out.extend_from_slice(chunk);
        out.extend_from_slice(&png[iend..]);
        out
    }

    /// One `iTXt` chunk, framed by `png` itself rather than by hand, ready to splice anywhere.
    fn framed(chunk: &ITXtChunk) -> Vec<u8> {
        use png::text_metadata::EncodableTextChunk;

        let mut encoded = Vec::new();
        chunk.encode(&mut encoded).expect("the chunk encodes");
        encoded
    }

    // ---------------------------------------------------------------------
    // The tail after IDAT
    // ---------------------------------------------------------------------

    #[test]
    fn a_workflow_chunk_after_the_image_data_is_still_found() {
        // `read_info` stops at the first IDAT, so before the second pass existed this file read as
        // Ok(None) — a silent "no recipe here" about a file that carries one. We never write the
        // chunk out here, but sc-15949 reads files from strangers and the placement is legal.
        let text = minimal_envelope_json("a lighthouse behind the image data");
        let spliced = splice_before_iend(&png_with(&[]), &framed(&workflow_chunk(text)));
        let share = read_workflow_chunk(&spliced)
            .expect("a post-IDAT chunk must not be an error")
            .expect("and must be found");
        assert_eq!(share.prompt, "a lighthouse behind the image data");

        // The streaming entry point reads the same file the same way — the second pass must not be
        // an artifact of having the whole slice in memory.
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("post-idat.png");
        std::fs::write(&path, &spliced).expect("writes the fixture");
        let from_file = read_workflow_chunk_file(&path)
            .expect("reads from the path")
            .expect("carries a workflow");
        assert_eq!(from_file, share);
    }

    #[test]
    fn a_pre_idat_chunk_wins_over_one_appended_after_the_image_data() {
        // The precedence rule, pinned. The pre-IDAT chunk is the one our encoder wrote in the same
        // pass that wrote the pixels; a post-IDAT one is something appended afterwards. So a file
        // carrying both resolves to the pre-IDAT recipe rather than reporting an ambiguity, and the
        // second pass never runs for a file of ours.
        let spliced = splice_before_iend(
            &png_with_workflow_text(&minimal_envelope_json("the recipe that made this image")),
            &framed(&workflow_chunk(minimal_envelope_json(
                "appended afterwards",
            ))),
        );
        let share = read_workflow_chunk(&spliced)
            .expect("reads")
            .expect("carries a workflow");
        assert_eq!(share.prompt, "the recipe that made this image");
    }

    #[test]
    fn duplicate_workflow_chunks_after_the_image_data_are_refused_too() {
        // The duplicate rule is the second pass's as much as the first's: two recipes in the tail
        // is the same ambiguity as two in the metadata, and must not resolve to whichever came
        // first.
        let mut tail = framed(&workflow_chunk(minimal_envelope_json("one")));
        tail.extend_from_slice(&framed(&workflow_chunk(minimal_envelope_json("two"))));
        assert_eq!(
            read_workflow_chunk(&splice_before_iend(&png_with(&[]), &tail)),
            Err(WorkflowChunkError::DuplicateChunk { count: 2 })
        );
    }

    #[test]
    fn a_foreign_keyword_after_the_image_data_still_reads_as_none() {
        // The second pass must not become a route by which any text chunk counts. A ComfyUI export
        // whose chunks were relocated past the image data is still an absence, not a workflow.
        let mut tail = framed(&ITXtChunk::new("parameters", "steps: 28, sampler: euler"));
        tail.extend_from_slice(&framed(&ITXtChunk::new(
            "Sceneworks:Workflow",
            "case matters",
        )));
        assert_eq!(
            read_workflow_chunk(&splice_before_iend(&png_with(&[]), &tail)),
            Ok(None)
        );
    }

    #[test]
    fn an_oversized_chunk_after_the_image_data_is_bounded_by_the_budget_and_never_read() {
        // The metadata budget has to apply out here too, or the tail would be an unbounded place to
        // put a chunk that the front of the file refuses. Over MAX_METADATA_BYTES of real data,
        // uncompressed so no ratio is involved and the size on disk is the size in memory.
        let huge = "A".repeat(MAX_METADATA_BYTES + 1);
        let tail = framed(&ITXtChunk::new(WORKFLOW_CHUNK_KEYWORD, huge));
        let result = read_workflow_chunk(&splice_before_iend(&png_with(&[]), &tail));

        // Ok(None) is the recorded decision — a budget spent in the tail is an absence, not an
        // error (`workflow_chunk_after_image_data` documents why), so this must not be Ok(Some) and
        // must not be an error either.
        assert_eq!(
            result,
            Ok(None),
            "an oversized post-IDAT chunk must be an absence, not a workflow and not an error"
        );
        // What Ok(None) proves is that the text reached neither the text cap nor the parser. Had the
        // budget let the chunk through, `select_workflow_chunk` would have matched our keyword and
        // the 8 MiB of text would have come back as TextTooLarge — which is exactly what the
        // neighbouring test with a 1 MiB chunk (under the metadata budget, over the text cap) gets,
        // and what this case itself returns once the decoder budget is widened to 64 MiB
        // (TextTooLarge { bytes: 8388609 }). So `png` refused it while framing.
        //
        // It does NOT prove the chunk was never buffered — it was. `png` grows a buffer for the
        // declared length and only then finds it over budget: measured, this case peaks at 8,408,413
        // bytes live, including one 8,388,709-byte allocation. The property is that the cost follows
        // MAX_METADATA_BYTES and not the header, so a 64 MiB declared chunk peaks at the same figure
        // as a 16 MiB one.
    }

    /// Roughly `mib` MiB of uncompressed `iTXt` under a foreign keyword — the fat but legitimate
    /// metadata a third-party writer might carry, here only to draw the metadata budget down.
    fn filler_chunk(name: &str, mib: usize) -> ITXtChunk {
        ITXtChunk::new(format!("sceneworks:{name}"), "A".repeat(mib * 1024 * 1024))
    }

    #[test]
    fn a_budget_spent_reaching_the_tail_is_an_absence_not_an_error() {
        // The cumulative-budget interaction, pinned as a decision rather than left to accident.
        // `png`'s `Limits::reserve_bytes` only ever subtracts, so MAX_METADATA_BYTES is one pool for
        // the whole file and pass one can leave too little of it to frame one more chunk out in the
        // tail. Before the tail pass existed, every file below read as Ok(None); surfacing the
        // exhausted budget as MetadataTooLarge would have turned case (c) into a user-visible error
        // on import (sc-15949), which is what the module contract forbids.
        let envelope = minimal_envelope_json("behind the image data");

        // (a) The PRIMARY walk survives 3 MiB of foreign pre-IDAT text: a workflow chunk in the
        //     metadata proper alongside that much filler still reads. So whatever (c) does, it is
        //     not because this much filler defeats pass one.
        let beside_the_filler = png_with(&[
            filler_chunk("filler-pre", 3),
            workflow_chunk(envelope.clone()),
        ]);
        assert_eq!(
            read_workflow_chunk(&beside_the_filler)
                .expect("the primary walk survives 3 MiB of foreign text")
                .expect("and finds the workflow beside it")
                .prompt,
            "behind the image data"
        );

        // (b) Straddling with room to spare: filler pre-IDAT, more filler in the tail, our chunk
        //     behind both. The tail stays reachable through a file that carries pre-IDAT metadata.
        let base = png_with(&[filler_chunk("filler-pre", 1)]);
        let mut tail = framed(&filler_chunk("filler-tail", 1));
        tail.extend_from_slice(&framed(&workflow_chunk(envelope.clone())));
        assert_eq!(
            read_workflow_chunk(&splice_before_iend(&base, &tail))
                .expect("a straddling file within the budget must not be an error")
                .expect("and its tail workflow must be found")
                .prompt,
            "behind the image data"
        );

        // (c) The same shape with fatter filler on both sides of the image data. Pass one succeeds
        //     — (a) proves 3 MiB does not defeat it — and then the tail walk runs out of pool before
        //     it reaches our chunk. The answer is the absence pass one already established, NOT an
        //     error: we are not being asked to read this metadata, only looking in one more place.
        let base = png_with(&[filler_chunk("filler-pre", 3)]);
        let mut tail = framed(&filler_chunk("filler-tail", 3));
        tail.extend_from_slice(&framed(&workflow_chunk(envelope)));
        assert_eq!(
            read_workflow_chunk(&splice_before_iend(&base, &tail)),
            Ok(None),
            "a budget exhausted on the way to the tail must read as an absence, not MetadataTooLarge"
        );
    }

    #[test]
    fn a_zip_bomb_after_the_image_data_refuses_instead_of_inflating() {
        // The decompression cap is the other guard that must not be bypassed by placement. 8 MiB of
        // NUL in a small chunk: nothing before the inflate can refuse it, so if the second pass
        // reached a laxer decompress this would cost 8 MiB for a few kB of theirs.
        let bomb = "\0".repeat(8 * 1024 * 1024);
        let tail = framed(&workflow_chunk(bomb));
        let spliced = splice_before_iend(&png_with(&[]), &tail);
        assert!(
            spliced.len() < 64 * 1024,
            "the bomb must stay small on disk or it is not testing the ratio: {} bytes",
            spliced.len()
        );
        match read_workflow_chunk(&spliced) {
            Err(WorkflowChunkError::UnreadableChunk { limit, .. }) => {
                assert_eq!(limit, MAX_WORKFLOW_TEXT_BYTES);
            }
            other => panic!("the post-IDAT bomb must be refused, got {other:?}"),
        }
    }

    #[test]
    fn an_uncompressed_chunk_over_the_cap_after_the_image_data_is_refused_too() {
        // Between the text cap and the metadata budget, written with the compression flag clear —
        // so it is the TEXT cap that has to catch it, on the second pass as on the first.
        let text = "A".repeat(MAX_WORKFLOW_TEXT_BYTES + 1);
        let tail = framed(&ITXtChunk::new(WORKFLOW_CHUNK_KEYWORD, text));
        assert_eq!(
            read_workflow_chunk(&splice_before_iend(&png_with(&[]), &tail)),
            Err(WorkflowChunkError::TextTooLarge {
                bytes: MAX_WORKFLOW_TEXT_BYTES + 1,
                limit: MAX_WORKFLOW_TEXT_BYTES
            })
        );
    }

    #[test]
    fn a_lying_chunk_length_after_the_image_data_refuses_without_allocating_it() {
        // The declared-length attack, moved into the tail. 0x7FFF_FFFF is the largest length a PNG
        // chunk header may declare; the file is a few hundred bytes. A reader that trusted the
        // header out here would ask for 2 GiB.
        let plain = png_with(&[]);
        for declared in [0x7FFF_FFFF, u32::MAX] {
            let liar = framed_chunk(
                b"iTXt",
                &itxt_data(WORKFLOW_CHUNK_KEYWORD, false, b"{}"),
                Some(declared),
            );
            // Either a typed refusal or a clean absence is correct; recovering a workflow from it
            // is not, and neither is allocating what it claims.
            let result = read_workflow_chunk(&splice_before_iend(&plain, &liar));
            assert!(
                !matches!(result, Ok(Some(_))),
                "a post-IDAT chunk declaring {declared} bytes must never yield a workflow, got \
                 {result:?}"
            );
        }
    }

    #[test]
    fn a_damaged_tail_is_an_absence_rather_than_an_error() {
        // Pass one already established there is no workflow in the metadata; a truncated or corrupt
        // tail cannot change that answer. Turning "this PNG is damaged and has no recipe" into a
        // hard error would break the Ok(None) promise for files that read cleanly today, so the
        // second pass swallows its own walk failures.
        let full = png_with(&[]);
        // Cut inside the image data, past every pre-IDAT chunk: there is no IEND to walk to.
        let truncated = &full[..full.len() - 6];
        assert_eq!(
            read_workflow_chunk(truncated),
            Ok(None),
            "a file with no workflow and a truncated tail must stay a clean absence"
        );

        // And a byte flipped in the tail must not turn the absence into a failure either.
        let mut corrupt = full.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xFF;
        assert_eq!(read_workflow_chunk(&corrupt), Ok(None));
    }

    #[test]
    fn the_hand_built_framing_agrees_with_the_encoder() {
        // Without this the adversarial tests below could all be "passing" because the spliced
        // chunk was malformed in some way the reader discards, rather than because the guard under
        // test fired. Here the hand-framed chunk is a legitimate one, so it must READ.
        let text = minimal_envelope_json("a lighthouse");
        let plain = png_with(&[]);
        let spliced = splice_after_ihdr(
            &plain,
            &framed_chunk(
                b"iTXt",
                &itxt_data(WORKFLOW_CHUNK_KEYWORD, true, &zlib_stored(text.as_bytes())),
                None,
            ),
        );
        let share = read_workflow_chunk(&spliced)
            .expect("the hand-framed chunk must be well formed")
            .expect("and must carry a workflow");
        assert_eq!(share.prompt, "a lighthouse");
    }

    // ---------------------------------------------------------------------
    // The quiet paths: no chunk is not an error
    // ---------------------------------------------------------------------

    #[test]
    fn a_png_without_a_workflow_chunk_reads_as_none() {
        assert_eq!(read_workflow_chunk(&png_with(&[])), Ok(None));
    }

    #[test]
    fn the_none_write_path_reads_back_as_none() {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("plain.png");
        write_workflow_chunk(&rgb_fixture(), &path, None).expect("writes");
        assert_eq!(read_workflow_chunk_file(&path), Ok(None));
    }

    #[test]
    fn a_chunk_under_another_keyword_reads_as_none() {
        // What a ComfyUI or Automatic1111 export looks like to us: text chunks, none of them ours.
        let foreign = [
            ITXtChunk::new("parameters", "steps: 28, sampler: euler"),
            ITXtChunk::new("prompt", "{\"1\":{\"class_type\":\"KSampler\"}}"),
            ITXtChunk::new(
                "sceneworks:workflow ",
                "trailing space is a different keyword",
            ),
            ITXtChunk::new("Sceneworks:Workflow", "case matters"),
        ];
        assert_eq!(read_workflow_chunk(&png_with(&foreign)), Ok(None));
    }

    // ---------------------------------------------------------------------
    // Strip (sc-15953)
    // ---------------------------------------------------------------------

    #[test]
    fn stripping_removes_the_chunk_and_leaves_the_rest_of_the_file_alone() {
        let embedded = png_with_workflow_text(&minimal_envelope_json("a lighthouse"));
        let plain = png_with(&[]);
        let stripped = strip_workflow_chunk(&embedded)
            .expect("an embedded PNG strips")
            .expect("and reports that something was taken out");

        // The read side agrees the chunk is gone…
        assert_eq!(read_workflow_chunk(&stripped), Ok(None));
        // …and what is left is the same file the None arm writes for the same pixels, byte for
        // byte. That is the pixel-identity proof: IHDR and every IDAT are the original's own bytes.
        assert!(
            stripped == plain,
            "the stripped file is {} bytes and the plain one {}",
            stripped.len(),
            plain.len()
        );
    }

    #[test]
    fn a_png_with_nothing_of_ours_in_it_reports_nothing_to_strip() {
        // Ok(None) is "use the source unchanged", which is what keeps the strip path from
        // rewriting a file that never carried a chunk. A foreign writer's text chunks stay put:
        // stripping ours is not a licence to scrub someone else's metadata out of their image.
        let foreign = png_with(&[
            ITXtChunk::new("parameters", "steps: 28, sampler: euler"),
            ITXtChunk::new("prompt", "{\"1\":{\"class_type\":\"KSampler\"}}"),
            ITXtChunk::new("Sceneworks:Workflow", "case matters"),
        ]);
        assert_eq!(strip_workflow_chunk(&foreign), Ok(None));
        assert_eq!(strip_workflow_chunk(&png_with(&[])), Ok(None));
    }

    #[test]
    fn a_file_that_is_not_a_png_has_nothing_to_strip() {
        // The write path only ever writes into a PNG, so a JPEG or an MP4 cannot be carrying one of
        // ours. "Save without the workflow" on a video is a plain copy, not an error.
        for input in [
            b"".as_slice(),
            b"\x89PNG".as_slice(),
            b"\xff\xd8\xff\xe0JFIF".as_slice(),
            b"{\"sceneworksWorkflow\":\"image\"}".as_slice(),
        ] {
            assert_eq!(
                strip_workflow_chunk(input),
                Ok(None),
                "{} bytes of non-PNG must be a clean no-op",
                input.len()
            );
        }
    }

    #[test]
    fn a_png_whose_framing_cannot_be_walked_refuses_rather_than_returning_the_original() {
        // The privacy-critical failure mode. Ok(None) means "use the source unchanged", so
        // answering it for a file we could not walk would hand the user a copy that may still carry
        // the chunk while telling them it does not. A truncated file must be an error.
        let full = png_with_workflow_text(&minimal_envelope_json("a lighthouse"));
        let truncated = &full[..full.len() - 6];
        assert!(
            matches!(
                strip_workflow_chunk(truncated),
                Err(WorkflowChunkError::Png { .. })
            ),
            "a truncated PNG must refuse, got {:?}",
            strip_workflow_chunk(truncated)
        );
        // And a chunk header claiming more than the file holds, which is the same class through a
        // different door.
        let liar = splice_after_ihdr(
            &png_with(&[]),
            &framed_chunk(
                b"iTXt",
                &itxt_data(WORKFLOW_CHUNK_KEYWORD, false, b"{}"),
                Some(0x7FFF_FFFF),
            ),
        );
        assert!(matches!(
            strip_workflow_chunk(&liar),
            Err(WorkflowChunkError::Png { .. })
        ));
    }

    /// A workflow chunk whose declared length is a lie, so the walker cannot step over it — the
    /// shape that hides one in a region no chunk walk can account for. Uncompressed, so the
    /// envelope text sits in the file as plain bytes a `grep` would find.
    fn unwalkable_hidden_chunk(envelope: &str) -> Vec<u8> {
        framed_chunk(
            b"iTXt",
            &itxt_data(WORKFLOW_CHUNK_KEYWORD, false, envelope.as_bytes()),
            Some(0x7FFF_FFFF),
        )
    }

    /// Whether `bytes` carry `needle` anywhere — a raw byte search, which is what a stranger's
    /// tooling does and what the reader's IEND-stopping walk does NOT do.
    fn byte_search(bytes: &[u8], needle: &[u8]) -> bool {
        bytes.len() >= needle.len() && bytes.windows(needle.len()).any(|window| window == needle)
    }

    #[test]
    fn a_workflow_chunk_hiding_past_iend_refuses_instead_of_reporting_success() {
        // Two reproduced failures, both of which falsified the invariant three lines above this
        // module's `strip_workflow_chunk`: "a caller asking for a stripped copy must refuse rather
        // than hand over the original".
        //
        // The escape hatch was the post-IEND copy-through. Trailing bytes past IEND are not chunks
        // under the spec, so a tail that cannot be walked was copied through verbatim to let a file
        // with a foreign tail survive the round trip. But that also copied through anything hidden
        // in it — and our own reader stops at IEND, so it reads the result back as clean. Only a
        // raw byte search finds the leftover, which is precisely the threat model: the file has
        // left the machine and it is a stranger's tooling doing the looking.
        let envelope = minimal_envelope_json("the prompt that must not leave");
        let secret = b"the prompt that must not leave";
        let hidden = unwalkable_hidden_chunk(&envelope);

        // (1) TAIL ONLY. Reproduced at the head this fixes as `Ok(None)` — which every caller reads
        //     as "use the source unchanged" — with the envelope still in the served bytes.
        let mut tail_only = png_with(&[]);
        tail_only.extend_from_slice(&hidden);
        assert!(
            byte_search(&tail_only, secret),
            "the fixture must actually carry the secret or it proves nothing"
        );
        assert!(
            read_workflow_chunk(&tail_only) == Ok(None),
            "our own reader stops at IEND, so it reports this file as clean — which is why the \
             guard cannot be built out of what the reader can see"
        );
        assert!(
            matches!(
                strip_workflow_chunk(&tail_only),
                Err(WorkflowChunkError::Png { .. })
            ),
            "a workflow chunk in an unwalkable tail must REFUSE, not answer Ok(None): got {:?}",
            strip_workflow_chunk(&tail_only)
        );

        // (2) A STRIPPABLE CHUNK PRE-IDAT AND A HIDDEN COPY IN THE TAIL. The worse one: reproduced
        //     as `Ok(Some(…))`, i.e. "I removed one", with the keyword AND the prompt still in the
        //     bytes handed back. A caller cannot tell that answer from a clean strip.
        let mut both = png_with_workflow_text(&envelope);
        both.extend_from_slice(&hidden);
        let stripped = strip_workflow_chunk(&both);
        assert!(
            matches!(stripped, Err(WorkflowChunkError::Png { .. })),
            "a file with one strippable chunk and one hidden past IEND must refuse rather than \
             report a removal it did not fully make: got {stripped:?}"
        );
        // And say so about the right thing, so the log line names the tail rather than reading as
        // a generic "corrupt PNG".
        let Err(WorkflowChunkError::Png { detail }) = stripped else {
            unreachable!("asserted above")
        };
        assert!(
            detail.contains(WORKFLOW_CHUNK_KEYWORD) && detail.contains("trailing"),
            "the refusal must name what it found and where: {detail:?}"
        );

        // (3) The copy helper is the seam a user actually touches, and it reported `Ok(true)` for
        //     case (2) while writing a file that still contained the prompt. It must refuse, and
        //     it must not leave a destination behind for the user to share by mistake.
        let directory = tempfile::tempdir().expect("temp dir");
        let source = directory.path().join("hidden.png");
        let destination = directory.path().join("shared.png");
        std::fs::write(&source, &both).expect("writes the fixture");
        assert!(
            matches!(
                copy_without_workflow_chunk(&source, &destination),
                Err(WorkflowChunkError::Png { .. })
            ),
            "the copy helper must not report success over a file it cannot fully clean"
        );
        assert!(
            !destination.exists(),
            "a refused strip must leave no file behind — a half-cleaned copy on disk is the thing \
             the user goes on to share"
        );
    }

    #[test]
    fn a_benign_unwalkable_tail_still_rides_through_the_strip() {
        // The other half of the guard above, and the half that stops it becoming "refuse any file
        // with a tail". Bytes past IEND that are not chunk framing and carry nothing of ours are
        // still copied through verbatim, so a file some other tool appended to survives the round
        // trip with its tail intact.
        let envelope = minimal_envelope_json("a lighthouse");
        let tail = b"\x00\x01not chunk framing, just bytes someone appended";
        let mut file = png_with_workflow_text(&envelope);
        file.extend_from_slice(tail);

        let stripped = strip_workflow_chunk(&file)
            .expect("a benign tail must not be an error")
            .expect("and the pre-IDAT chunk is still removed");
        let mut expected = png_with(&[]);
        expected.extend_from_slice(tail);
        assert!(
            stripped == expected,
            "the tail must survive byte for byte alongside the excision"
        );
        assert_eq!(read_workflow_chunk(&stripped), Ok(None));
    }

    #[test]
    fn a_png_whose_framing_ends_without_an_iend_refuses() {
        // Reproduced at the head this fixes as `Ok(Some(…))`: every chunk walked cleanly, our chunk
        // was excised, and the caller got a PNG with no IEND in it. "Save a copy without the
        // workflow" then wrote an undecodable file and reported success — the strip was honest and
        // the artifact was junk, which is its own way of failing the user.
        let full = png_with_workflow_text(&minimal_envelope_json("a lighthouse"));
        let no_iend = &full[..iend_offset(&full)];
        // The cut is at a chunk boundary, so nothing about the framing is malformed — this is not
        // the pre-existing "runs past the end of the file" case wearing a different hat.
        assert_eq!(
            chunk_end(no_iend, AFTER_IHDR),
            Some(AFTER_IHDR + framed_workflow_chunk_len(&minimal_envelope_json("a lighthouse"))),
            "the fixture must be cut at a chunk boundary or it tests the wrong guard"
        );
        let result = strip_workflow_chunk(no_iend);
        assert!(
            matches!(result, Err(WorkflowChunkError::Png { .. })),
            "a PNG whose framing ends without an IEND must refuse: got {result:?}"
        );
        let Err(WorkflowChunkError::Png { detail }) = result else {
            unreachable!("asserted above")
        };
        assert!(
            detail.contains("IEND"),
            "the refusal must name the missing IEND: {detail:?}"
        );

        // And a file with NOTHING of ours in it is refused on the same ground rather than quietly
        // answered Ok(None) — the promise is about the walk, not about what the walk found.
        assert!(matches!(
            strip_workflow_chunk(&png_with(&[])[..iend_offset(&png_with(&[]))]),
            Err(WorkflowChunkError::Png { .. })
        ));
    }

    #[test]
    fn the_kept_spans_are_the_complement_of_the_removed_ones() {
        // The API's download route slices the file it already read by these spans instead of
        // copying the kept bytes into a second buffer, so an off-by-one here is a corrupt download
        // rather than a failed test. Pinned against the buffer-building path, which cannot drift
        // from it because both go through `kept_spans`.
        let embedded = png_with_workflow_text(&minimal_envelope_json("a lighthouse"));
        let spans = workflow_chunk_spans(&embedded).expect("walks");
        assert_eq!(spans.len(), 1, "one chunk in, one span out");

        let mut assembled = Vec::new();
        for kept in kept_spans(embedded.len(), &spans) {
            assembled.extend_from_slice(&embedded[kept.start..kept.end]);
        }
        assert!(
            assembled
                == strip_workflow_chunk(&embedded)
                    .expect("strips")
                    .expect("had something to strip"),
            "assembling the kept spans must reproduce the stripped buffer exactly"
        );
        assert!(assembled == png_with(&[]));

        // Degenerate shapes, so the helper is not only exercised through one happy case.
        assert_eq!(kept_spans(10, &[]), vec![Span { start: 0, end: 10 }]);
        assert_eq!(kept_spans(10, &[Span { start: 0, end: 10 }]), vec![]);
        assert_eq!(
            kept_spans(10, &[Span { start: 0, end: 4 }, Span { start: 6, end: 8 }]),
            vec![Span { start: 4, end: 6 }, Span { start: 8, end: 10 }]
        );
    }

    #[test]
    fn stripping_reaches_every_place_a_chunk_can_hide() {
        // Post-IDAT is where sc-15949's foreign writers put theirs and where the reader's second
        // pass looks, so the stripper has to go there too — a strip that only cleared the metadata
        // proper would leave a readable workflow in the file it just promised had none.
        let envelope = minimal_envelope_json("behind the image data");
        let tail = splice_before_iend(&png_with(&[]), &framed(&workflow_chunk(envelope.clone())));
        assert!(read_workflow_chunk(&tail).expect("reads").is_some());
        let stripped = strip_workflow_chunk(&tail)
            .expect("strips")
            .expect("had something to strip");
        assert_eq!(read_workflow_chunk(&stripped), Ok(None));
        assert!(
            stripped == png_with(&[]),
            "the tail chunk was excised whole"
        );

        // Both at once — the duplicate case that reads as an error rather than a workflow. A file
        // the reader refuses is still a file with our prompt in it, so it must strip clean.
        let both = splice_before_iend(
            &png_with_workflow_text(&envelope),
            &framed(&workflow_chunk(envelope.clone())),
        );
        assert_eq!(
            read_workflow_chunk(
                &strip_workflow_chunk(&both)
                    .expect("strips")
                    .expect("had something to strip")
            ),
            Ok(None)
        );

        // And under the other two text-chunk types, which our writer never emits and our reader
        // never reads — the stripper is broader than both on purpose.
        for kind in [b"tEXt", b"zTXt"] {
            let mut data = WORKFLOW_CHUNK_KEYWORD.as_bytes().to_vec();
            data.push(0);
            data.extend_from_slice(b"whatever a foreign writer put here");
            let spliced = splice_after_ihdr(&png_with(&[]), &framed_chunk(kind, &data, None));
            let stripped = strip_workflow_chunk(&spliced)
                .expect("strips")
                .unwrap_or_else(|| {
                    panic!(
                        "{} under our keyword must be taken out",
                        String::from_utf8_lossy(kind)
                    )
                });
            assert!(stripped == png_with(&[]));
        }
    }

    #[test]
    fn the_copy_helper_strips_or_copies_and_says_which() {
        let directory = tempfile::tempdir().expect("temp dir");
        let source = directory.path().join("embedded.png");
        let destination = directory.path().join("shared.png");
        let envelope =
            parse_workflow_share_json(&minimal_envelope_json("a lighthouse")).expect("parses");

        write_workflow_chunk(&rgb_fixture(), &source, Some(&envelope)).expect("writes");
        assert!(read_workflow_chunk_file(&source).expect("reads").is_some());
        assert_eq!(
            copy_without_workflow_chunk(&source, &destination),
            Ok(true),
            "the source carried a chunk, so the copy reports removing one"
        );
        assert_eq!(read_workflow_chunk_file(&destination), Ok(None));

        // The source is untouched: turning the switch off, or saving one copy without the
        // workflow, changes nothing about files already written.
        assert!(read_workflow_chunk_file(&source).expect("reads").is_some());

        // A source with nothing to strip goes through `std::fs::copy`, so the plain and the
        // without-workflow save produce the same file.
        let plain = directory.path().join("plain.png");
        let plain_copy = directory.path().join("plain-copy.png");
        write_workflow_chunk(&rgb_fixture(), &plain, None).expect("writes");
        assert_eq!(copy_without_workflow_chunk(&plain, &plain_copy), Ok(false));
        assert_eq!(
            std::fs::read(&plain).expect("reads"),
            std::fs::read(&plain_copy).expect("reads")
        );
    }

    // ---------------------------------------------------------------------
    // Adversarial corpus
    // ---------------------------------------------------------------------

    #[test]
    fn non_png_inputs_are_typed_errors() {
        for input in [
            b"".as_slice(),
            b"\x89PNG".as_slice(),
            b"{\"sceneworksWorkflow\":\"image\"}".as_slice(),
            b"\xff\xd8\xff\xe0JFIF".as_slice(),
            &[0_u8; 4096],
        ] {
            assert_eq!(
                read_workflow_chunk(input),
                Err(WorkflowChunkError::NotPng),
                "input of {} bytes must be reported as not-a-PNG",
                input.len()
            );
        }
    }

    /// Framed size of the compressed chunk our writer emits for `text`.
    fn framed_workflow_chunk_len(text: &str) -> usize {
        use png::text_metadata::EncodableTextChunk;

        let mut encoded = Vec::new();
        workflow_chunk(text.to_owned())
            .encode(&mut encoded)
            .expect("the fixture chunk encodes");
        encoded.len()
    }

    #[test]
    fn a_truncated_png_never_yields_a_partial_workflow() {
        let text = minimal_envelope_json("a lighthouse");
        let full = png_with_workflow_text(&text);
        let intact = read_workflow_chunk(&full)
            .expect("the intact file reads")
            .expect("and carries a workflow");
        // Our chunk sits between IHDR and IDAT, so this is the offset one past its CRC.
        let chunk_end = AFTER_IHDR + framed_workflow_chunk_len(&text);

        // Every truncation, not a sampled few: the interesting cuts are mid-length-field,
        // mid-keyword and mid-zlib-stream, and which byte offsets those land on is an encoder
        // detail nobody should have to keep in sync.
        for cut in 1..full.len() {
            match read_workflow_chunk(&full[..cut]) {
                Ok(None) | Err(_) => {}
                Ok(Some(share)) => {
                    // A cut PAST the chunk legitimately still reads: the metadata is complete and
                    // CRC-checked, and only the pixel stream is missing. What must never happen is
                    // a workflow recovered from an incomplete chunk, or a DIFFERENT one recovered
                    // from a damaged file.
                    assert!(
                        cut > chunk_end,
                        "a cut at {cut} lands inside the chunk (ends at {chunk_end}) yet produced \
                         a workflow"
                    );
                    assert_eq!(
                        share, intact,
                        "a truncated PNG produced a different workflow at {cut}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_corrupt_png_never_panics() {
        let full = png_with_workflow_text(&minimal_envelope_json("a lighthouse"));
        for offset in 8..full.len() {
            let mut corrupt = full.clone();
            corrupt[offset] ^= 0xFF;
            // The only contract is "returns": Ok(None), Ok(Some) (a flipped bit in the IDAT
            // stream leaves our chunk intact) and Err are all acceptable outcomes.
            let _ = read_workflow_chunk(&corrupt);
        }
    }

    #[test]
    fn a_lying_chunk_length_refuses_without_allocating_it() {
        // 0x7FFF_FFFF is the largest length a PNG chunk header may declare. The file is a few
        // hundred bytes; a reader that trusted the header would ask for 2 GiB.
        let plain = png_with(&[]);
        let liar = framed_chunk(
            b"iTXt",
            &itxt_data(WORKFLOW_CHUNK_KEYWORD, false, b"{}"),
            Some(0x7FFF_FFFF),
        );
        let spliced = splice_after_ihdr(&plain, &liar);
        assert!(
            read_workflow_chunk(&spliced).is_err(),
            "a chunk claiming 2 GiB in a 200-byte file must be refused"
        );

        // And a length beyond what the spec allows at all.
        let beyond = framed_chunk(
            b"iTXt",
            &itxt_data(WORKFLOW_CHUNK_KEYWORD, false, b"{}"),
            Some(u32::MAX),
        );
        assert!(read_workflow_chunk(&splice_after_ihdr(&plain, &beyond)).is_err());
    }

    #[test]
    fn an_oversized_chunk_is_bounded_by_the_budget_and_refused() {
        // An honestly-declared but absurd chunk: over MAX_METADATA_BYTES of real data, written
        // UNCOMPRESSED so the size on disk is the size in memory and no ratio is involved.
        //
        // The refusal does not precede the allocation. `png` grows a buffer for the declared length
        // and only then finds it over budget — measured, the equivalent post-IDAT case peaks at
        // 8,408,413 bytes live. What MAX_METADATA_BYTES buys is that the peak follows OUR number
        // rather than the header's, which is what makes a 2 GiB claim (the test above) cheap.
        let huge = "A".repeat(MAX_METADATA_BYTES + 1);
        let png = png_with(&[ITXtChunk::new(WORKFLOW_CHUNK_KEYWORD, huge)]);
        assert_eq!(
            read_workflow_chunk(&png),
            Err(WorkflowChunkError::MetadataTooLarge {
                limit: MAX_METADATA_BYTES
            })
        );
    }

    #[test]
    fn a_zip_bomb_refuses_instead_of_inflating() {
        // 8 MiB of NUL, which deflate takes to a handful of kilobytes: a ~1000:1 ratio. The file
        // is small, so nothing before the inflate can refuse it — the decompression cap is the
        // only thing standing between this and 8 MiB of our memory for a few kB of theirs.
        let bomb = "\0".repeat(8 * 1024 * 1024);
        let png = png_with_workflow_text(&bomb);
        assert!(
            png.len() < 64 * 1024,
            "the bomb must stay small on disk or it is not testing the ratio: {} bytes",
            png.len()
        );
        match read_workflow_chunk(&png) {
            Err(WorkflowChunkError::UnreadableChunk { limit, .. }) => {
                assert_eq!(limit, MAX_WORKFLOW_TEXT_BYTES);
            }
            other => panic!("the bomb must be refused, got {other:?}"),
        }
    }

    #[test]
    fn text_one_byte_under_the_cap_still_decompresses() {
        // The other half of the bomb test, and the half that makes it mean something: a payload
        // just under the cap must get all the way THROUGH decompression, so the refusal above is
        // the threshold firing and not a blanket "compressed text is suspicious". It then fails as
        // an ENVELOPE error, which is proof it reached the parser.
        let padding = "A".repeat(MAX_WORKFLOW_TEXT_BYTES - 16);
        let text = format!("{{\"pad\":\"{padding}\"}}");
        assert!(text.len() <= MAX_WORKFLOW_TEXT_BYTES);
        assert_eq!(
            read_workflow_chunk(&png_with_workflow_text(&text)),
            Err(WorkflowChunkError::Envelope(
                WorkflowShareError::MissingMarker
            )),
            "a payload under the cap must reach the envelope parser"
        );

        // One byte over, and it is refused by the cap instead.
        let over = format!("{{\"pad\":\"{}\"}}", "A".repeat(MAX_WORKFLOW_TEXT_BYTES));
        assert!(over.len() > MAX_WORKFLOW_TEXT_BYTES);
        match read_workflow_chunk(&png_with_workflow_text(&over)) {
            Err(WorkflowChunkError::UnreadableChunk { limit, .. }) => {
                assert_eq!(limit, MAX_WORKFLOW_TEXT_BYTES);
            }
            other => panic!("a payload over the cap must be refused, got {other:?}"),
        }
    }

    #[test]
    fn an_uncompressed_chunk_over_the_cap_is_refused_too() {
        // The compression flag is the attacker's choice, so the cap cannot live only on the
        // inflate path. Between the text cap and the metadata budget, so it is the text cap that
        // must catch it.
        let text = "A".repeat(MAX_WORKFLOW_TEXT_BYTES + 1);
        let png = png_with(&[ITXtChunk::new(WORKFLOW_CHUNK_KEYWORD, text)]);
        assert_eq!(
            read_workflow_chunk(&png),
            Err(WorkflowChunkError::TextTooLarge {
                bytes: MAX_WORKFLOW_TEXT_BYTES + 1,
                limit: MAX_WORKFLOW_TEXT_BYTES
            })
        );
    }

    #[test]
    fn duplicate_workflow_chunks_are_refused_not_resolved() {
        let first = minimal_envelope_json("the recipe that made this image");
        let second = minimal_envelope_json("a recipe someone appended afterwards");
        let png = png_with(&[
            workflow_chunk(first),
            workflow_chunk(second.clone()),
            // A foreign keyword alongside must not count toward the duplicate tally.
            ITXtChunk::new("parameters", "steps: 28"),
        ]);
        assert_eq!(
            read_workflow_chunk(&png),
            Err(WorkflowChunkError::DuplicateChunk { count: 2 })
        );

        // Three of them, to prove the count is real and not a hardcoded 2.
        let png = png_with(&[
            workflow_chunk(minimal_envelope_json("one")),
            workflow_chunk(minimal_envelope_json("two")),
            workflow_chunk(second),
        ]);
        assert_eq!(
            read_workflow_chunk(&png),
            Err(WorkflowChunkError::DuplicateChunk { count: 3 })
        );
    }

    #[test]
    fn a_chunk_that_is_not_utf8_is_handled_both_compressed_and_not() {
        let plain = png_with(&[]);
        // A lone 0x80 continuation byte, and a truncated 3-byte sequence.
        let not_utf8: &[u8] = &[0x7B, 0x80, 0xE2, 0x82];

        // Compressed: the bytes survive framing and fail on the way out of the inflater, so this
        // is our error to return.
        let compressed = splice_after_ihdr(
            &plain,
            &framed_chunk(
                b"iTXt",
                &itxt_data(WORKFLOW_CHUNK_KEYWORD, true, &zlib_stored(not_utf8)),
                None,
            ),
        );
        match read_workflow_chunk(&compressed) {
            Err(WorkflowChunkError::UnreadableChunk { .. }) => {}
            other => panic!("non-UTF-8 compressed text must be refused, got {other:?}"),
        }

        // Uncompressed: `png` rejects the chunk while framing it and, `iTXt` being ancillary,
        // drops it rather than failing the whole file. So the workflow is simply absent — a typed
        // absence, not a panic and not a lossy decode.
        let uncompressed = splice_after_ihdr(
            &plain,
            &framed_chunk(
                b"iTXt",
                &itxt_data(WORKFLOW_CHUNK_KEYWORD, false, not_utf8),
                None,
            ),
        );
        assert_eq!(read_workflow_chunk(&uncompressed), Ok(None));
    }

    #[test]
    fn a_corrupt_compressed_payload_is_a_typed_error() {
        let plain = png_with(&[]);
        for payload in [
            b"not a zlib stream at all".as_slice(),
            // A valid zlib header followed by garbage.
            &[0x78, 0x01, 0xFF, 0xFF, 0xFF, 0xFF],
            // A stored block whose Adler-32 trailer is wrong.
            &{
                let mut stream = zlib_stored(b"{\"a\":1}");
                let last = stream.len() - 1;
                stream[last] ^= 0xFF;
                stream
            },
            b"".as_slice(),
        ] {
            let png = splice_after_ihdr(
                &plain,
                &framed_chunk(
                    b"iTXt",
                    &itxt_data(WORKFLOW_CHUNK_KEYWORD, true, payload),
                    None,
                ),
            );
            match read_workflow_chunk(&png) {
                Err(WorkflowChunkError::UnreadableChunk { .. }) => {}
                other => panic!("a corrupt payload must be a typed error, got {other:?}"),
            }
        }
    }

    #[test]
    fn valid_json_of_the_wrong_shape_is_an_envelope_error() {
        let cases: &[(&str, WorkflowShareError)] = &[
            ("[1, 2, 3]", WorkflowShareError::NotAnObject),
            ("\"just a string\"", WorkflowShareError::NotAnObject),
            ("null", WorkflowShareError::NotAnObject),
            ("{\"steps\": 28}", WorkflowShareError::MissingMarker),
            (
                "{\"sceneworksWorkflow\": \"image\"}",
                WorkflowShareError::MissingSchemaVersion,
            ),
            (
                "{\"sceneworksWorkflow\": \"hologram\", \"schemaVersion\": 1}",
                WorkflowShareError::UnsupportedKind {
                    found: "hologram".to_owned(),
                    supported: "image, video".to_owned(),
                },
            ),
            (
                "{\"sceneworksWorkflow\": \"image\", \"schemaVersion\": 99}",
                WorkflowShareError::UnsupportedSchemaVersion {
                    file: 99,
                    supported: 1,
                },
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(
                read_workflow_chunk(&png_with_workflow_text(text)),
                Err(WorkflowChunkError::Envelope(expected.clone())),
                "for {text}"
            );
        }

        // And text that is not JSON at all.
        match read_workflow_chunk(&png_with_workflow_text("not json, just prose")) {
            Err(WorkflowChunkError::Envelope(WorkflowShareError::InvalidJson(_))) => {}
            other => panic!("non-JSON text must be an envelope error, got {other:?}"),
        }
    }

    #[test]
    fn the_message_for_every_error_names_the_problem() {
        // These strings reach a user through sc-15952, so each must read as a sentence rather than
        // a debug dump. Cheap to assert, and it stops a new variant shipping with an empty arm.
        let messages = [
            WorkflowChunkError::Io {
                detail: "access denied".to_owned(),
            },
            WorkflowChunkError::NotPng,
            WorkflowChunkError::Png {
                detail: "unexpected end of file".to_owned(),
            },
            WorkflowChunkError::DuplicateChunk { count: 2 },
            WorkflowChunkError::MetadataTooLarge { limit: 1 },
            WorkflowChunkError::TextTooLarge { bytes: 2, limit: 1 },
            WorkflowChunkError::UnreadableChunk {
                limit: 1,
                detail: "needs more space".to_owned(),
            },
            WorkflowChunkError::Envelope(WorkflowShareError::MissingMarker),
            WorkflowChunkError::Encode {
                detail: "zero width".to_owned(),
            },
        ];
        for error in messages {
            let message = error.to_string();
            assert!(
                message.len() > 20 && message.ends_with('.'),
                "{error:?} renders as {message:?}"
            );
        }
    }
}
