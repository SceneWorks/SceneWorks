import { describe, expect, it } from "vitest";

import {
  SOURCE_WORKFLOW_ABSENT,
  SOURCE_WORKFLOW_PENDING,
  SOURCE_WORKFLOW_PRESENT,
  SOURCE_WORKFLOW_UNKNOWN,
  describeOpenedImage,
  originalExportFilename,
  workflowStateFromInspect,
} from "./editorSourceWorkflow.js";
import {
  MAX_WORKFLOW_INSPECT_BYTES,
  WORKFLOW_STATUS_NO_WORKFLOW,
  WORKFLOW_STATUS_WORKFLOW,
} from "./workflowShare.js";

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

// jsdom's `Blob` has no `arrayBuffer()`, so the stand-in provides the two methods the module
// actually uses. `size` is settable independently of the bytes, which is the whole point: the
// module's size branch has to be exercisable without allocating half a gigabyte.
function blobOf(bytes, { size = bytes.length } = {}) {
  return {
    size,
    arrayBuffer: async () => Uint8Array.from(bytes).buffer,
    slice: (start, end) => blobOf(Array.from(bytes).slice(start, end), { size: end - start }),
  };
}

const PNG = blobOf([...SIGNATURE, 1, 2, 3, 4]);

// ---------------------------------------------------------------------------
// The mapping from the one reader's answer to what the editor shows
// ---------------------------------------------------------------------------
//
// The Rust side of this pairing is `the_readers_verdict_over_the_whole_corpus` in
// `crates/sceneworks-core/tests/workflow_png.rs`, which runs the 17-file corpus through
// `read_workflow_chunk` and pins each file to one of three buckets. This pins what the editor does
// with each bucket. Together they are the whole claim: the editor's verdict IS the reader's, for
// every file in the corpus, because nothing here re-derives it.
describe("the reader's verdict, as an editor state (sc-15954)", () => {
  it("maps the endpoint's two success shapes", () => {
    expect(workflowStateFromInspect({ status: WORKFLOW_STATUS_WORKFLOW })).toBe(
      SOURCE_WORKFLOW_PRESENT,
    );
    expect(workflowStateFromInspect({ status: WORKFLOW_STATUS_NO_WORKFLOW })).toBe(
      SOURCE_WORKFLOW_ABSENT,
    );
  });

  // The corpus rows the reader REFUSES — malformed JSON, a future schemaVersion, a non-image
  // marker kind, oversized text, two chunks, a truncated file — all arrive as a thrown ApiError,
  // and the caller maps a throw to `unknown`. What must never happen is any of them arriving as
  // `absent`: the editor would then drop a real recipe on an edit and say nothing.
  it("never reads an unrecognized body as 'no recipe'", () => {
    for (const body of [null, undefined, {}, { status: "" }, { status: "unreadable" }, { workflow: null }]) {
      expect(workflowStateFromInspect(body)).toBe(SOURCE_WORKFLOW_UNKNOWN);
    }
  });
});

describe("describing the file the editor opened (sc-15954)", () => {
  it("queues a PNG for the reader rather than answering itself", async () => {
    const described = await describeOpenedImage(PNG, "portrait.png");
    expect(described).toMatchObject({
      filename: "portrait.png",
      workflow: SOURCE_WORKFLOW_PENDING,
    });
    expect(described.blob).toBe(PNG);
  });

  // Null means "re-rasterize", which is what the editor did for every source before this story.
  // A JPEG cannot carry the envelope, so nothing is lost by re-encoding it — and the pre-existing
  // "open a JPEG, Download a PNG" behaviour is left exactly as it was.
  it("declines a non-PNG source", async () => {
    expect(await describeOpenedImage(blobOf([0xff, 0xd8, 0xff, 1, 2, 3, 4, 5]), "photo.jpg")).toBeNull();
    expect(await describeOpenedImage(blobOf([0x89, 0x50]), "stub.png")).toBeNull();
    expect(await describeOpenedImage(null, "gone.png")).toBeNull();
  });

  it("declines a blob it cannot read rather than guessing", async () => {
    const unreadable = {
      size: 64,
      slice: () => unreadable,
      arrayBuffer: async () => {
        throw new Error("gone");
      },
    };
    expect(await describeOpenedImage(unreadable, "gone.png")).toBeNull();
  });

  // THE regression this replacement closes. The deleted chunk walker gave up past a 64 MiB scan
  // ceiling and reported `carriesWorkflow: false` — indistinguishable from "this file has no
  // recipe" — for an 8192² upscale output, which is a routine artifact and not a corner case.
  // There is no size ceiling below the endpoint's own now, so such a file is measured.
  it("still asks the reader about a file far past the deleted 64 MiB scan ceiling", async () => {
    const huge = blobOf([...SIGNATURE, 1, 2], { size: 200 * 1024 * 1024 });
    expect(await describeOpenedImage(huge, "upscale.png")).toMatchObject({
      workflow: SOURCE_WORKFLOW_PENDING,
    });
  });

  // Past the ENDPOINT's cap there is genuinely no way to ask: it stages the whole body to disk
  // before looking at a byte and then returns 413. So the file is not uploaded — and the state is
  // `unknown`, which renders a note, rather than `absent`, which would read as "no recipe".
  it("says unknown — never absent — for a PNG past the endpoint's own cap", async () => {
    const over = blobOf([...SIGNATURE, 1, 2], { size: MAX_WORKFLOW_INSPECT_BYTES + 1 });
    const described = await describeOpenedImage(over, "enormous.png");
    expect(described).toMatchObject({
      filename: "enormous.png",
      workflow: SOURCE_WORKFLOW_UNKNOWN,
    });
    // The passthrough is unaffected: those bytes still go out whole, chunk and all.
    expect(described.blob).toBe(over);
  });

  it("names the passthrough after the file, never '-edited'", () => {
    expect(originalExportFilename("portrait.png")).toBe("portrait.png");
    expect(originalExportFilename("Untitled layout")).toBe("Untitled layout.png");
    expect(originalExportFilename("")).toBe("image.png");
    expect(originalExportFilename(null)).toBe("image.png");
  });
});
