import { describe, expect, it } from "vitest";

import {
  MAX_WORKFLOW_SCAN_BYTES,
  WORKFLOW_CHUNK_KEYWORD,
  carriesReadableWorkflow,
  countWorkflowChunks,
  describeOpenedImage,
  isPngBytes,
  originalExportFilename,
} from "./pngWorkflowChunk.js";

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function ascii(text) {
  return [...text].map((character) => character.charCodeAt(0));
}

// One framed PNG chunk: length | type | data | CRC. The CRC is zeroed — the walker under test
// reads lengths and types and never verifies one, exactly like the Rust `workflow_chunk_spans`
// this mirrors, so a real CRC would make the fixture longer and the test no stronger.
function chunk(kind, data) {
  const length = data.length;
  return [
    (length >>> 24) & 0xff,
    (length >>> 16) & 0xff,
    (length >>> 8) & 0xff,
    length & 0xff,
    ...ascii(kind),
    ...data,
    0,
    0,
    0,
    0,
  ];
}

// An `iTXt` payload: keyword NUL | compression flag | method | language NUL | translated NUL | text
function itxtData(keyword, text = '{"sceneworksWorkflow":"image"}') {
  return [...ascii(keyword), 0, 0, 0, 0, 0, ...ascii(text)];
}

function png(...chunks) {
  return Uint8Array.from([...SIGNATURE, ...chunks.flat()]);
}

const IHDR = chunk("IHDR", new Array(13).fill(0));
const IEND = chunk("IEND", []);
const WORKFLOW_ITXT = chunk("iTXt", itxtData(WORKFLOW_CHUNK_KEYWORD));

describe("PNG workflow-chunk detection (sc-15954)", () => {
  it("recognizes the signature and rejects anything else", () => {
    expect(isPngBytes(png(IHDR, IEND))).toBe(true);
    expect(isPngBytes(Uint8Array.from(ascii("\xff\xd8\xffnot a png")))).toBe(false);
    expect(isPngBytes(Uint8Array.from([0x89, 0x50]))).toBe(false);
    expect(isPngBytes(null)).toBe(false);
  });

  it("finds one workflow iTXt chunk and reports the file as carrying a recipe", () => {
    const bytes = png(IHDR, WORKFLOW_ITXT, IEND);
    expect(countWorkflowChunks(bytes)).toBe(1);
    expect(carriesReadableWorkflow(bytes)).toBe(true);
  });

  it("says no for an ordinary PNG with no chunk of ours", () => {
    const bytes = png(IHDR, chunk("iTXt", itxtData("parameters")), IEND);
    expect(countWorkflowChunks(bytes)).toBe(0);
    expect(carriesReadableWorkflow(bytes)).toBe(false);
  });

  // The reader is `read_workflow_chunk`, which refuses a file carrying two matching chunks
  // (`DuplicateChunk`: which of two recipes made the image is not a question the file answers).
  // Printing "Recipe included" over a file nobody can read a recipe out of would be exactly the
  // over-claim this epic keeps finding.
  it("does not claim a recipe travels when the file carries two of them", () => {
    const bytes = png(IHDR, WORKFLOW_ITXT, WORKFLOW_ITXT, IEND);
    expect(countWorkflowChunks(bytes)).toBe(2);
    expect(carriesReadableWorkflow(bytes)).toBe(false);
  });

  // The stripper matches tEXt/zTXt too, deliberately — a removal that leaves a copy behind is
  // worse than one that removes too much. The READER does not, and this is the reader's claim.
  it("ignores a tEXt spelling of the keyword, which no reader would decode", () => {
    const bytes = png(IHDR, chunk("tEXt", [...ascii(WORKFLOW_CHUNK_KEYWORD), 0, 0x7b]), IEND);
    expect(carriesReadableWorkflow(bytes)).toBe(false);
  });

  it("stops at IEND, so bytes appended past the terminator never count", () => {
    const bytes = png(IHDR, IEND, WORKFLOW_ITXT);
    expect(countWorkflowChunks(bytes)).toBe(0);
  });

  it("refuses to over-count on framing it cannot walk", () => {
    // A chunk declaring 4 GiB of payload in a 100-byte file: the length must not be read as
    // negative (which would walk the cursor backwards) nor trusted (which would read past the
    // end). Either way the answer is "no recipe".
    const lying = [0xff, 0xff, 0xff, 0xff, ...ascii("iTXt"), 0x00];
    expect(countWorkflowChunks(png(IHDR, lying, WORKFLOW_ITXT, IEND))).toBe(0);
    // Truncated mid-chunk.
    expect(countWorkflowChunks(png(IHDR, WORKFLOW_ITXT.slice(0, 6)))).toBe(0);
    // Not a PNG at all.
    expect(countWorkflowChunks(Uint8Array.from(ascii("GIF89a")))).toBe(0);
    expect(countWorkflowChunks(Uint8Array.from([]))).toBe(0);
  });

  it("does not match a keyword that merely starts with ours", () => {
    const bytes = png(IHDR, chunk("iTXt", itxtData(`${WORKFLOW_CHUNK_KEYWORD}2`)), IEND);
    expect(carriesReadableWorkflow(bytes)).toBe(false);
  });
});

describe("describing the file the editor opened (sc-15954)", () => {
  function blobOf(bytes, { size = bytes.length } = {}) {
    return {
      size,
      arrayBuffer: async () => Uint8Array.from(bytes).buffer,
      slice: (start, end) => blobOf(Array.from(bytes).slice(start, end), { size: end - start }),
    };
  }

  it("reports a PNG carrying a recipe, under the original name", async () => {
    const described = await describeOpenedImage(
      blobOf(png(IHDR, WORKFLOW_ITXT, IEND)),
      "portrait.png",
    );
    expect(described).toMatchObject({ filename: "portrait.png", carriesWorkflow: true });
  });

  it("reports a PNG with no recipe, still eligible for the byte-exact passthrough", async () => {
    const described = await describeOpenedImage(blobOf(png(IHDR, IEND)), "scan.png");
    expect(described).toMatchObject({ filename: "scan.png", carriesWorkflow: false });
  });

  // Null means "re-rasterize", which is what the editor did for every source before this story.
  // A JPEG cannot carry the envelope, so nothing is lost by re-encoding it — and the pre-existing
  // "open a JPEG, Download a PNG" behaviour is left exactly as it was.
  it("declines a non-PNG source", async () => {
    expect(await describeOpenedImage(blobOf(ascii("\xff\xd8\xffjpeg")), "photo.jpg")).toBeNull();
  });

  it("declines a blob it cannot read rather than guessing", async () => {
    const unreadable = {
      size: 10,
      arrayBuffer: async () => {
        throw new Error("gone");
      },
    };
    expect(await describeOpenedImage(unreadable, "gone.png")).toBeNull();
    expect(await describeOpenedImage(null, "gone.png")).toBeNull();
  });

  // Past the scan ceiling the signature alone decides: the file still passes through byte for
  // byte (which is the branch that preserves anything it carries), and the header simply does not
  // claim a recipe it declined to measure.
  it("keeps the passthrough but claims no recipe for a file too large to scan", async () => {
    const bytes = png(IHDR, WORKFLOW_ITXT, IEND);
    const described = await describeOpenedImage(
      blobOf(bytes, { size: MAX_WORKFLOW_SCAN_BYTES + 1 }),
      "huge.png",
    );
    expect(described).toMatchObject({ filename: "huge.png", carriesWorkflow: false });
    expect(described.blob.size).toBe(MAX_WORKFLOW_SCAN_BYTES + 1);
  });

  it("names the passthrough after the file, never '-edited'", () => {
    expect(originalExportFilename("portrait.png")).toBe("portrait.png");
    expect(originalExportFilename("Untitled layout")).toBe("Untitled layout.png");
    expect(originalExportFilename("")).toBe("image.png");
    expect(originalExportFilename(null)).toBe("image.png");
  });
});
