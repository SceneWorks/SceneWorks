// Does a PNG in the browser's hands carry a workflow a RECIPIENT can read? (sc-15954, epic 15945)
//
// The Image Editor is the one egress path that does not hand a file through unchanged, so it is
// the one path that has to say what its download will contain. That sentence is only worth
// printing if it is measured rather than assumed, which is what this module is: a byte-level walk
// of the PNG chunk framing, with no decode and no allocation past the input.
//
// # Why this mirrors the READER and not the stripper
//
// `crates/sceneworks-core/src/workflow_png.rs` has two keyword matchers, and they deliberately
// disagree:
//
// * `read_workflow_chunk` — what a recipient actually gets — reads `png::Info::utf8_text`, which
//   is `iTXt` and nothing else, and REFUSES a file carrying more than one matching chunk
//   (`WorkflowChunkError::DuplicateChunk`: which of two recipes made the image is not a question
//   the file answers).
// * `workflow_chunk_spans` — the stripper — matches `tEXt` / `zTXt` / `iTXt` alike, because a
//   removal that leaves a copy behind is worse than one that removes too much.
//
// This is the claim "the recipe travels with this copy", so it follows the READER: exactly one
// `iTXt` under the keyword. A `tEXt` that spells the keyword is not something anyone can read
// back, and neither is a pair of `iTXt`s, so neither counts as a recipe here.
//
// # What a false answer costs, in each direction
//
// A false POSITIVE is the one that matters: it prints "Recipe included" over a file that carries
// none, which is the class of defect this epic exists to prevent. So every path that cannot be
// walked cleanly answers `false` — a truncated file, a chunk length that runs off the end, a
// keyword with no terminator. A false NEGATIVE costs an "edited, so the recipe is not carried"
// note that is not shown; the export is unchanged either way.

/// The `iTXt` keyword the envelope lives under. The Rust `WORKFLOW_CHUNK_KEYWORD`.
export const WORKFLOW_CHUNK_KEYWORD = "sceneworks:workflow";

/// The 8-byte PNG signature.
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/// `iTXt`, as the four type bytes.
const ITXT_TYPE = [0x69, 0x54, 0x58, 0x74];
/// `IEND`. The terminator: a decoder stops here, so anything past it is not readable metadata.
const IEND_TYPE = [0x49, 0x45, 0x4e, 0x44];

/// Length + type + CRC around a chunk's payload.
const CHUNK_OVERHEAD = 12;

/// PNG bounds a text keyword at 79 bytes; ours is 19, so a NUL further out than this is not our
/// chunk and the scan does not have to read the rest of the payload to know it.
const MAX_KEYWORD_BYTES = 79;

const KEYWORD_BYTES = Uint8Array.from(
  WORKFLOW_CHUNK_KEYWORD.split("").map((character) => character.charCodeAt(0)),
);

function matchesAt(bytes, offset, expected) {
  for (let i = 0; i < expected.length; i += 1) {
    if (bytes[offset + i] !== expected[i]) return false;
  }
  return true;
}

/// Whether `bytes` starts with the PNG signature. Eight bytes is enough — callers that only have
/// the head of a large file pass just those.
export function isPngBytes(bytes) {
  if (!bytes || bytes.length < PNG_SIGNATURE.length) return false;
  return matchesAt(bytes, 0, PNG_SIGNATURE);
}

/// Big-endian u32 at `offset`, or null when it runs past the end.
function readU32(bytes, offset) {
  if (offset + 4 > bytes.length) return null;
  // `>>> 0` keeps a length with the high bit set unsigned rather than negative — a hostile file
  // can declare one, and a negative length would walk the cursor backwards forever.
  return (
    ((bytes[offset] << 24) |
      (bytes[offset + 1] << 16) |
      (bytes[offset + 2] << 8) |
      bytes[offset + 3]) >>>
    0
  );
}

/// Whether a chunk payload's keyword (everything up to its first NUL) is ours.
function keywordIsOurs(bytes, dataStart, dataLength) {
  if (dataLength < KEYWORD_BYTES.length + 1) return false;
  const limit = Math.min(dataLength, MAX_KEYWORD_BYTES + 1);
  let terminator = -1;
  for (let i = 0; i < limit; i += 1) {
    if (bytes[dataStart + i] === 0) {
      terminator = i;
      break;
    }
  }
  if (terminator !== KEYWORD_BYTES.length) return false;
  return matchesAt(bytes, dataStart, KEYWORD_BYTES);
}

/// How many readable `sceneworks:workflow` `iTXt` chunks `bytes` carries, stopping at `IEND`.
///
/// Returns 0 for anything that is not a walkable PNG. The walk stops at the first chunk header it
/// cannot trust, so a file whose framing goes bad halfway reports only what was found BEFORE that
/// point — the same direction the reader errs in, and never an over-count.
export function countWorkflowChunks(bytes) {
  if (!isPngBytes(bytes)) return 0;
  let cursor = PNG_SIGNATURE.length;
  let found = 0;
  while (cursor + 8 <= bytes.length) {
    const length = readU32(bytes, cursor);
    if (length === null) break;
    const dataStart = cursor + 8;
    // The declared payload plus its framing must fit. `length` is unsigned and can be up to
    // 2^32-1, so this is the bound that stops a declared-but-absent payload being read.
    const next = cursor + CHUNK_OVERHEAD + length;
    if (next > bytes.length) break;
    if (matchesAt(bytes, cursor + 4, ITXT_TYPE) && keywordIsOurs(bytes, dataStart, length)) {
      found += 1;
    }
    if (matchesAt(bytes, cursor + 4, IEND_TYPE)) break;
    cursor = next;
  }
  return found;
}

/// Whether a recipient opening these bytes would get a recipe out of them.
///
/// Exactly one chunk. Zero is the ordinary case for every image in the world; two or more is a
/// file `read_workflow_chunk` refuses, so claiming a recipe travels in it would be a lie of the
/// precise kind this epic keeps finding.
export function carriesReadableWorkflow(bytes) {
  return countWorkflowChunks(bytes) === 1;
}

/// Rough upper bound on what the editor will read into memory to answer the question above.
///
/// Past it the file is checked for the PNG signature alone (an 8-byte slice) and reported as
/// carrying no readable workflow — the honest answer for a file we declined to measure, and the
/// one that under-claims rather than over-claims.
export const MAX_WORKFLOW_SCAN_BYTES = 64 * 1024 * 1024;

/// The name an untouched PNG downloads under.
///
/// Deliberately NOT `editedFilename`'s `-edited` suffix: this branch hands back the file that was
/// opened, byte for byte, and calling it "edited" would be the same category of misstatement the
/// rest of this story is about. Always `.png`, because the branch only runs for a PNG source.
export function originalExportFilename(name) {
  const base = (name || "image").replace(/\.[^./\\]+$/, "").trim() || "image";
  return `${base}.png`;
}

/// Measure an opened blob: is it a PNG we can hand back unchanged, and does it carry a recipe?
///
/// Returns null when the answer is "re-rasterize it" — a non-PNG, or a blob that could not be
/// read. Null is the pre-sc-15954 behaviour, so every failure degrades to what the editor did
/// before rather than to a wrong claim.
export async function describeOpenedImage(blob, name, { maxScanBytes = MAX_WORKFLOW_SCAN_BYTES } = {}) {
  if (!blob || typeof blob.arrayBuffer !== "function") return null;
  let png = false;
  let carriesWorkflow = false;
  try {
    if (blob.size <= maxScanBytes) {
      const bytes = new Uint8Array(await blob.arrayBuffer());
      png = isPngBytes(bytes);
      carriesWorkflow = png && carriesReadableWorkflow(bytes);
    } else {
      const head = new Uint8Array(await blob.slice(0, PNG_SIGNATURE.length).arrayBuffer());
      png = isPngBytes(head);
    }
  } catch {
    return null;
  }
  if (!png) return null;
  return { blob, filename: originalExportFilename(name), carriesWorkflow };
}
