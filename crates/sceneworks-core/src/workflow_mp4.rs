//! The MP4 metadata codec for the sanitized workflow envelope (sc-15956, epic 15945).
//!
//! [`crate::workflow_share`] owns the *contract*; this module owns the MP4 *plumbing*, exactly as
//! [`crate::workflow_png`] owns the PNG plumbing. The two containers could not be less alike, and
//! every difference below is a measured one rather than a preference.
//!
//! # The measurement that shaped this module
//!
//! MP4 has no `iTXt`. It has `moov/udta/meta/ilst`, a tag store whose well-known keys are
//! four-character codes, and ffmpeg — which is what writes every MP4 in this tree — exposes it as
//! named metadata. Three routes were measured on a real ffmpeg before any of this was written, with
//! a full-cap (160 kB) envelope, and two of the three do not work:
//!
//! | route | result |
//! |-------|--------|
//! | `-metadata sceneworks_workflow=…`, a CUSTOM tag | written only with `-movflags use_metadata_tags`, and **dropped by every downstream tool that does not pass that same flag** — including a plain `-c copy` remux |
//! | `-metadata comment=…` on the COMMAND LINE | `ENAMETOOLONG` above ~32 kB. Windows caps a command line at 32,767 characters and the envelope ceiling is 160 KiB |
//! | `-f ffmetadata -i <file>` + `-map_metadata`, tag `comment` | survives every transcode measured, at the full cap, byte-exact |
//!
//! So: the **standard `comment` tag**, written from a **file**. Not a namespaced key, which is the
//! opposite of the choice [`crate::workflow_png::WORKFLOW_CHUNK_KEYWORD`] makes and for a reason
//! that only applies here — a custom MP4 tag is not merely unconventional, it is *lossy*, and a
//! namespace nobody can read is worth less than a shared key we can identify by its contents.
//! [`WORKFLOW_MP4_TAG`] says the rest.
//!
//! Identification therefore cannot lean on the key. It leans on the marker inside the JSON, the
//! same [`crate::workflow_share::WORKFLOW_SHARE_MARKER_KEY`] the PNG side publishes under: a
//! `comment` that is not our envelope is somebody else's comment, and [`read_workflow_metadata`]
//! returns `Ok(None)` for it rather than an error. That is the common case for every MP4 the user
//! did not generate here.
//!
//! # What a re-encode does NOT survive, and why the tag is still worth writing
//!
//! `-map_metadata -1` — an explicit strip — removes it, as does any pipeline that rebuilds the
//! container from streams alone. A platform that re-encodes an upload is such a pipeline. So the
//! sharing premise that holds for a PNG chunk (the bytes travel intact) holds for MP4 only where
//! the *file* travels: a copy, a download, a pass-through upload. sc-15956 records the measured
//! matrix in full. What that leaves is still the larger half of the feature — every generated clip
//! carries its own recipe, locally and through any file-preserving hop — and it is why the marker
//! kind exists, so a reader can tell a video envelope from an image one.
//!
//! # The read side is a trust boundary, and MP4s are large
//!
//! The PNG reader can afford `std::fs::read`; a video cannot — a 4K clip is gigabytes and the tag
//! is in a few hundred bytes of header. [`read_workflow_metadata_file`] therefore walks the box
//! tree over a seeking reader, skipping `mdat` without touching it, and every bound it applies is
//! ours rather than the file's:
//!
//! * a box that declares a size larger than the file is refused, not trusted;
//! * [`MAX_WORKFLOW_TEXT_BYTES`] caps the payload actually read out of a `data` box, so a header
//!   claiming 2 GiB costs a bounded read and a typed error;
//! * [`MAX_BOX_DEPTH`] bounds nesting, so a hand-built file of ten thousand nested `udta` boxes
//!   terminates;
//! * [`MAX_BOXES_SCANNED`] bounds the total walk, so a file of a million empty boxes does too.
//!
//! Parsing goes through [`crate::workflow_share::parse_workflow_share_json`] and nothing else, so
//! the sc-15946 allow-list sanitizer runs on the way in exactly as it does for a PNG.

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::workflow_share::{
    parse_workflow_share_json, WorkflowShare, WorkflowShareError, WORKFLOW_KIND_VIDEO,
    WORKFLOW_SHARE_MARKER_KEY,
};

/// The MP4 tag the envelope is published under: the standard `comment`, which ffmpeg maps to the
/// `moov/udta/meta/ilst/©cmt` atom.
///
/// **Not namespaced, unlike the PNG keyword, and that is the measured choice rather than a
/// concession.** A custom tag (`sceneworks_workflow`) is writable — ffmpeg's mov muxer emits
/// arbitrary keys under `-movflags use_metadata_tags` — but the mov muxer only *re-emits* unknown
/// keys when that same flag is passed again, so the tag is dropped by any tool that does not know
/// to ask for it. Measured: it does not survive a plain `-c copy` remux. `comment` survives
/// everything short of a deliberate strip.
///
/// The cost is that the key is shared with whatever else writes a comment, and the mitigation is
/// that the key was never what identified the envelope: [`WORKFLOW_SHARE_MARKER_KEY`] inside the
/// JSON is, in this container exactly as in a PNG.
pub const WORKFLOW_MP4_TAG: &str = "comment";

/// The first line of an `ffmetadata` document. ffmpeg refuses the file without it.
pub const FFMETADATA_HEADER: &str = ";FFMETADATA1";

/// Hard cap on the text read out of one `data` box, in bytes. 1 MiB.
///
/// The same number and the same derivation as [`crate::workflow_png::MAX_WORKFLOW_TEXT_BYTES`] —
/// the envelope contract's own worst case is ~150 kB and the serialized whole is refused past
/// `WORKFLOW_SHARE_MAX_BYTES` = 160 KiB regardless. There is no decompression here (an `ilst` value
/// is stored plain), so this is a bound on a claimed length rather than on an expansion ratio: it
/// is what stops a `data` box header declaring 2 GiB from being believed.
pub const MAX_WORKFLOW_TEXT_BYTES: usize = 1024 * 1024;

/// How deep the box walk will descend. The path we want is 5 levels
/// (`moov/udta/meta/ilst/©cmt/data`), so this is ample and still finite.
const MAX_BOX_DEPTH: usize = 8;

/// How many boxes the walk will look at before giving up. Bounds a file whose header is a million
/// empty boxes; a real MP4's header is a few hundred.
const MAX_BOXES_SCANNED: usize = 100_000;

/// The four-character codes on the path to the comment.
const BOX_MOOV: [u8; 4] = *b"moov";
const BOX_UDTA: [u8; 4] = *b"udta";
const BOX_META: [u8; 4] = *b"meta";
const BOX_ILST: [u8; 4] = *b"ilst";
const BOX_DATA: [u8; 4] = *b"data";
/// `©cmt` — the iTunes comment atom. `0xA9` is the copyright sign the four-character codes use as
/// their "free-form text" prefix; it is a byte, not a character, and writing it as one avoids any
/// question about the source file's encoding.
const BOX_CMT: [u8; 4] = [0xA9, b'c', b'm', b't'];

/// A `data` box's fixed prefix: 4 bytes of version+flags (1 = UTF-8 text) and 4 of locale.
const DATA_BOX_PREFIX: u64 = 8;

/// A box header: 4 bytes of size, 4 of type.
const BOX_HEADER: u64 = 8;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a workflow envelope could not be encoded for, or read out of, an MP4.
///
/// Every variant is a sentence a user can act on, for the same reason
/// [`crate::workflow_png::WorkflowChunkError`] is: these files travel, and "invalid box size at
/// offset 41264" is not an answer to "why won't this clip load its recipe?".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowMp4Error {
    /// Opening, reading or writing the file failed.
    Io { detail: String },
    /// The container's box framing is truncated, overlapping or otherwise unreadable.
    Malformed { detail: String },
    /// The comment atom is larger than [`MAX_WORKFLOW_TEXT_BYTES`].
    TooLarge { bytes: usize, limit: usize },
    /// The comment atom is not valid UTF-8, so it did not come from this app.
    NotUtf8,
    /// The envelope was found and refused by the contract. Carries the reason verbatim.
    Share(WorkflowShareError),
    /// The envelope contains a character `ffmetadata` cannot represent. Unreachable for a
    /// serialized [`WorkflowShare`] — see [`ffmetadata_document`] — and typed so that a future
    /// field that broke the assumption would fail loudly rather than silently truncate.
    Unencodable { detail: String },
}

impl fmt::Display for WorkflowMp4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { detail } => write!(formatter, "Could not read the video file: {detail}"),
            Self::Malformed { detail } => write!(
                formatter,
                "This video's container structure could not be read: {detail}"
            ),
            Self::TooLarge { bytes, limit } => write!(
                formatter,
                "This video's embedded workflow is {bytes} bytes, over the {limit}-byte limit."
            ),
            Self::NotUtf8 => write!(
                formatter,
                "This video's comment is not UTF-8 text, so it did not come from SceneWorks."
            ),
            Self::Share(error) => write!(formatter, "{error}"),
            Self::Unencodable { detail } => write!(
                formatter,
                "This workflow cannot be written into a video file: {detail}"
            ),
        }
    }
}

impl std::error::Error for WorkflowMp4Error {}

impl From<WorkflowShareError> for WorkflowMp4Error {
    fn from(error: WorkflowShareError) -> Self {
        Self::Share(error)
    }
}

fn io_error(error: &std::io::Error) -> WorkflowMp4Error {
    WorkflowMp4Error::Io {
        detail: error.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Write: the ffmetadata document
// ---------------------------------------------------------------------------

/// Escape one value for an `ffmetadata` document.
///
/// ffmpeg's own rule: `=`, `;`, `#`, `\` and a newline are escaped with a backslash. There is no
/// escape for a carriage return — see [`ffmetadata_document`], which is why this is not public.
fn escape_ffmetadata(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 16);
    for character in value.chars() {
        if matches!(character, '=' | ';' | '#' | '\\' | '\n') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

/// The `ffmetadata` document that puts `share` in the [`WORKFLOW_MP4_TAG`] tag.
///
/// # The carriage-return hazard
///
/// `ffmetadata`'s line parser has escapes for `=`, `;`, `#`, `\` and `\n`, and **none for `\r`**.
/// A raw carriage return in a value silently TRUNCATES it at that byte — measured: `"a\rb"` written,
/// `"a"` read back, no error from ffmpeg at either end. That is the worst failure shape available
/// (a recipe that is half a recipe and claims to be whole), so it is refused rather than escaped:
/// there is nothing to escape it *to*.
///
/// It is also unreachable. The value is `serde_json::to_string` output, and JSON escapes every
/// control character — a `\r` inside a prompt is the two bytes `\` `r`, never the one byte 0x0D.
/// The check below is therefore a guard on an invariant rather than a branch anyone takes, and it
/// exists because "unreachable" and "unchecked" are how a future field that stops serializing
/// through `serde_json` would corrupt every clip it touched without failing anything.
///
/// # Errors
/// [`WorkflowMp4Error::Unencodable`] if the serialized envelope contains a raw carriage return, and
/// [`WorkflowMp4Error::Share`] if it cannot be serialized at all.
pub fn ffmetadata_document(share: &WorkflowShare) -> Result<String, WorkflowMp4Error> {
    let payload = serde_json::to_string(share).map_err(|error| WorkflowMp4Error::Unencodable {
        detail: error.to_string(),
    })?;
    if payload.contains('\r') {
        return Err(WorkflowMp4Error::Unencodable {
            detail: "the envelope contains a carriage return, which `ffmetadata` truncates at \
                     rather than escapes"
                .to_owned(),
        });
    }
    Ok(format!(
        "{FFMETADATA_HEADER}\n{WORKFLOW_MP4_TAG}={}\n",
        escape_ffmetadata(&payload)
    ))
}

/// Write [`ffmetadata_document`] to `path`, for ffmpeg to read with `-f ffmetadata -i <path>`.
///
/// A FILE rather than a `-metadata` argument, and that is not a style choice: a command-line
/// argument is capped at 32,767 characters on Windows and the envelope ceiling is 160 KiB, so the
/// argv route fails with `ENAMETOOLONG` on any envelope past ~32 kB. Measured before this module
/// existed; see the module docs.
///
/// # Errors
/// [`WorkflowMp4Error::Io`] if the file cannot be written, plus [`ffmetadata_document`]'s.
pub fn write_workflow_metadata_file(
    share: &WorkflowShare,
    path: &Path,
) -> Result<(), WorkflowMp4Error> {
    let document = ffmetadata_document(share)?;
    std::fs::write(path, document).map_err(|error| io_error(&error))
}

/// The ffmpeg arguments that attach a written [`write_workflow_metadata_file`] document as an
/// input, and map it onto the output's container metadata.
///
/// `input_index` is the index this input will take among ffmpeg's `-i` arguments — the caller knows
/// it, because the caller built the rest of the command. Returned as two halves because they go in
/// different places: the input pair belongs with the other `-i`s, and `-map_metadata` belongs with
/// the output options.
#[must_use]
pub fn ffmetadata_input_args(path: &Path) -> Vec<String> {
    vec![
        "-f".to_owned(),
        "ffmetadata".to_owned(),
        "-i".to_owned(),
        path.to_string_lossy().into_owned(),
    ]
}

/// The `-map_metadata` pair naming the input [`ffmetadata_input_args`] added.
#[must_use]
pub fn ffmetadata_map_args(input_index: usize) -> Vec<String> {
    vec!["-map_metadata".to_owned(), input_index.to_string()]
}

/// How many bytes the envelope occupies in an MP4's tag store — the measurement the recording
/// ceiling is spent in for this container.
///
/// The `ilst` framing (`©cmt` + `data` headers + version/flags + locale) is 24 bytes on top of the
/// UTF-8 payload. Reported so a caller can compare containers honestly; the ceiling itself is
/// applied to the serialized envelope by `embeddable_workflow_share`, identically for both.
///
/// # Errors
/// [`WorkflowMp4Error::Unencodable`] if the envelope cannot be serialized.
pub fn workflow_metadata_size(share: &WorkflowShare) -> Result<usize, WorkflowMp4Error> {
    let payload = serde_json::to_string(share).map_err(|error| WorkflowMp4Error::Unencodable {
        detail: error.to_string(),
    })?;
    // `©cmt` header (8) + `data` header (8) + version/flags (4) + locale (4).
    Ok(payload.len() + 24)
}

// ---------------------------------------------------------------------------
// Read: the box walk
// ---------------------------------------------------------------------------

/// Read the envelope out of an MP4 on disk.
///
/// Seeks rather than reads: a generated clip is routinely hundreds of megabytes and the tag lives
/// in the header, so `mdat` is skipped over without ever being buffered.
///
/// `Ok(None)` for a video with no comment, or with a comment that is not one of ours — the common
/// case for every MP4 the user did not generate here, and not something to shout about.
///
/// # Errors
/// See [`WorkflowMp4Error`].
pub fn read_workflow_metadata_file(path: &Path) -> Result<Option<WorkflowShare>, WorkflowMp4Error> {
    let file = File::open(path).map_err(|error| io_error(&error))?;
    let length = file.metadata().map_err(|error| io_error(&error))?.len();
    let mut reader = BufReader::new(file);
    read_workflow_metadata_from(&mut reader, length)
}

/// [`read_workflow_metadata_file`] against bytes already in memory. For tests and for callers that
/// hold a small clip; prefer the file form for anything a user generated.
///
/// # Errors
/// See [`WorkflowMp4Error`].
pub fn read_workflow_metadata(bytes: &[u8]) -> Result<Option<WorkflowShare>, WorkflowMp4Error> {
    let length = bytes.len() as u64;
    let mut cursor = std::io::Cursor::new(bytes);
    read_workflow_metadata_from(&mut cursor, length)
}

/// The comment string an MP4 carries, before it is judged to be an envelope or not.
///
/// Separate from [`read_workflow_metadata`] so a caller that wants to know what is actually in the
/// tag — a diagnostic, a test — does not have to infer it from a `None`.
///
/// # Errors
/// See [`WorkflowMp4Error`].
pub fn read_mp4_comment(bytes: &[u8]) -> Result<Option<String>, WorkflowMp4Error> {
    let length = bytes.len() as u64;
    let mut cursor = std::io::Cursor::new(bytes);
    find_comment(&mut cursor, length)
}

fn read_workflow_metadata_from<R: Read + Seek>(
    reader: &mut R,
    length: u64,
) -> Result<Option<WorkflowShare>, WorkflowMp4Error> {
    let Some(comment) = find_comment(reader, length)? else {
        return Ok(None);
    };
    // The key is shared, so the marker inside decides. A comment somebody else wrote is an
    // absence, not a failure — the same answer a PNG with no `sceneworks:workflow` chunk gets.
    if !comment.contains(WORKFLOW_SHARE_MARKER_KEY) {
        return Ok(None);
    }
    let share = parse_workflow_share_json(&comment)?;
    // An MP4 carries a VIDEO workflow. Nothing else — the symmetrical half of the assertion
    // `workflow_png::read_workflow_chunk_from` makes, and for the same reason: the contract's kind
    // gate accepts every kind this build knows, which is right for an envelope and wrong for a
    // CONSUMER. Our writers only ever pair an MP4 with a video envelope, so a mismatch is a
    // hand-made file or a future build's, and a reader that surfaced an image recipe out of a clip
    // would be offering something no video studio can run.
    if share.kind != WORKFLOW_KIND_VIDEO {
        return Err(WorkflowMp4Error::Share(
            WorkflowShareError::UnsupportedKind {
                found: share.kind,
                supported: WORKFLOW_KIND_VIDEO.to_owned(),
            },
        ));
    }
    Ok(Some(share))
}

/// One box header, read at the reader's current position.
struct BoxHeader {
    kind: [u8; 4],
    /// Where the box's payload starts, absolute.
    body_start: u64,
    /// Where the box ends, absolute.
    end: u64,
}

/// Read a box header at `offset`, bounded by `limit` (the end of the enclosing box, or the file).
fn read_box_header<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    limit: u64,
) -> Result<Option<BoxHeader>, WorkflowMp4Error> {
    if offset.saturating_add(BOX_HEADER) > limit {
        return Ok(None);
    }
    reader
        .seek(SeekFrom::Start(offset))
        .map_err(|error| io_error(&error))?;
    let mut header = [0u8; 8];
    reader
        .read_exact(&mut header)
        .map_err(|error| io_error(&error))?;
    let declared = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let kind = [header[4], header[5], header[6], header[7]];

    let (body_start, size) = match declared {
        // `size == 1` means the real size is a 64-bit value after the type.
        1 => {
            if offset.saturating_add(16) > limit {
                return Err(WorkflowMp4Error::Malformed {
                    detail: "a 64-bit box size runs past the end of the file".to_owned(),
                });
            }
            let mut large = [0u8; 8];
            reader
                .read_exact(&mut large)
                .map_err(|error| io_error(&error))?;
            (offset + 16, u64::from_be_bytes(large))
        }
        // `size == 0` means "to the end of the enclosing box".
        0 => (offset + BOX_HEADER, limit - offset),
        _ => (offset + BOX_HEADER, u64::from(declared)),
    };

    let end = offset
        .checked_add(size)
        .ok_or(WorkflowMp4Error::Malformed {
            detail: "a box size overflows the file offset".to_owned(),
        })?;
    if size < body_start - offset || end > limit {
        return Err(WorkflowMp4Error::Malformed {
            detail: format!(
                "a `{}` box declares {size} bytes, which does not fit inside its parent",
                String::from_utf8_lossy(&kind)
            ),
        });
    }
    Ok(Some(BoxHeader {
        kind,
        body_start,
        end,
    }))
}

/// Walk `moov/udta/meta/ilst/©cmt/data` and return the comment text.
fn find_comment<R: Read + Seek>(
    reader: &mut R,
    length: u64,
) -> Result<Option<String>, WorkflowMp4Error> {
    let mut scanned = 0usize;
    descend(reader, 0, length, &PATH, 0, &mut scanned)
}

/// The containers on the way to the comment, in order. `meta` is handled specially (see
/// [`descend`]) because its FullBox prefix is not universal.
const PATH: [[u8; 4]; 4] = [BOX_MOOV, BOX_UDTA, BOX_META, BOX_ILST];

fn descend<R: Read + Seek>(
    reader: &mut R,
    start: u64,
    limit: u64,
    path: &[[u8; 4]],
    depth: usize,
    scanned: &mut usize,
) -> Result<Option<String>, WorkflowMp4Error> {
    if depth > MAX_BOX_DEPTH {
        return Ok(None);
    }
    let mut offset = start;
    while let Some(header) = read_box_header(reader, offset, limit)? {
        *scanned += 1;
        if *scanned > MAX_BOXES_SCANNED {
            return Err(WorkflowMp4Error::Malformed {
                detail: format!("more than {MAX_BOXES_SCANNED} boxes in the header"),
            });
        }
        // A zero-length box would spin the walk forever.
        if header.end <= offset {
            return Err(WorkflowMp4Error::Malformed {
                detail: "a box declares no length, so the walk cannot advance".to_owned(),
            });
        }
        if path.is_empty() {
            // Inside `ilst`: the comment atom, whose only child is a `data` box.
            if header.kind == BOX_CMT {
                if let Some(text) = read_comment_data(reader, header.body_start, header.end)? {
                    return Ok(Some(text));
                }
            }
        } else if header.kind == path[0] {
            // `meta` is a FullBox in the ISO/iTunes flavour ffmpeg writes (4 bytes of version and
            // flags before its children) and a plain container in some QuickTime files. Sniffing
            // is what makes the reader work on both: if the four bytes after the prefix are a
            // plausible box type, the prefix was really a box header and there is no FullBox.
            let body_start = if header.kind == BOX_META {
                meta_body_start(reader, header.body_start, header.end)?
            } else {
                header.body_start
            };
            if let Some(found) = descend(
                reader,
                body_start,
                header.end,
                &path[1..],
                depth + 1,
                scanned,
            )? {
                return Ok(Some(found));
            }
        }
        offset = header.end;
    }
    Ok(None)
}

/// Where a `meta` box's children start: after its 4-byte version+flags, unless the box is the
/// prefix-less QuickTime form.
fn meta_body_start<R: Read + Seek>(
    reader: &mut R,
    body_start: u64,
    end: u64,
) -> Result<u64, WorkflowMp4Error> {
    // A FullBox `meta` is followed by a child box header, so bytes 4..8 of its body are that
    // child's TYPE. A prefix-less `meta` starts with the child header immediately, so bytes 4..8
    // are the type and bytes 0..4 a size. Distinguishing them: read bytes 4..8 and ask whether they
    // look like a four-character code. `hdlr` does; the 0x00000000 version+flags word does not.
    if body_start.saturating_add(8) > end {
        return Ok(body_start);
    }
    reader
        .seek(SeekFrom::Start(body_start))
        .map_err(|error| io_error(&error))?;
    let mut probe = [0u8; 8];
    reader
        .read_exact(&mut probe)
        .map_err(|error| io_error(&error))?;
    if is_box_type(&probe[4..8]) {
        // bytes 0..4 were a size and 4..8 a type: no FullBox prefix.
        Ok(body_start)
    } else {
        Ok(body_start + 4)
    }
}

/// Whether four bytes look like a four-character box type: printable ASCII, or the `0xA9` prefix
/// the iTunes atoms use.
fn is_box_type(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .all(|byte| (0x20..=0x7E).contains(byte) || *byte == 0xA9)
}

/// Read the `data` box inside a `©cmt` atom.
fn read_comment_data<R: Read + Seek>(
    reader: &mut R,
    start: u64,
    limit: u64,
) -> Result<Option<String>, WorkflowMp4Error> {
    let mut offset = start;
    while let Some(header) = read_box_header(reader, offset, limit)? {
        if header.end <= offset {
            return Err(WorkflowMp4Error::Malformed {
                detail: "a `data` box declares no length".to_owned(),
            });
        }
        if header.kind == BOX_DATA {
            let payload_start = header.body_start + DATA_BOX_PREFIX;
            if payload_start > header.end {
                return Err(WorkflowMp4Error::Malformed {
                    detail: "a `data` box is too short to hold its own type indicator".to_owned(),
                });
            }
            let payload_len = header.end - payload_start;
            // OUR bound, not the file's: a header claiming 2 GiB costs this comparison.
            let payload_len =
                usize::try_from(payload_len).map_err(|_| WorkflowMp4Error::TooLarge {
                    bytes: usize::MAX,
                    limit: MAX_WORKFLOW_TEXT_BYTES,
                })?;
            if payload_len > MAX_WORKFLOW_TEXT_BYTES {
                return Err(WorkflowMp4Error::TooLarge {
                    bytes: payload_len,
                    limit: MAX_WORKFLOW_TEXT_BYTES,
                });
            }
            reader
                .seek(SeekFrom::Start(payload_start))
                .map_err(|error| io_error(&error))?;
            let mut payload = vec![0u8; payload_len];
            reader
                .read_exact(&mut payload)
                .map_err(|error| io_error(&error))?;
            return String::from_utf8(payload)
                .map(Some)
                .map_err(|_| WorkflowMp4Error::NotUtf8);
        }
        offset = header.end;
    }
    Ok(None)
}
