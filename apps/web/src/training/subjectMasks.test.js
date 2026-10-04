import { describe, expect, it } from "vitest";

import {
  SUBJECT_MASK_MAX_UPLOAD_BYTES,
  subjectMaskFileError,
  subjectMaskStatusByItemId,
  subjectMaskUrl,
} from "./subjectMasks.js";

// sc-2126 / epic 2123 E6: the editor's mask-upload limits match the API's
// (crates/sceneworks-core/src/training_subject_masks.rs: SUBJECT_MASK_MAX_UPLOAD_BYTES = 32 MiB,
// PNG/JPEG/WebP).
describe("subject mask upload limits", () => {
  it("accepts a PNG/JPEG/WebP at the size limit", () => {
    for (const type of ["image/png", "image/jpeg", "image/webp"]) {
      expect(subjectMaskFileError({ type, size: SUBJECT_MASK_MAX_UPLOAD_BYTES })).toBe("");
    }
    expect(SUBJECT_MASK_MAX_UPLOAD_BYTES).toBe(32 * 1024 * 1024);
  });

  it("refuses a missing file, an unsupported type, and one byte over the limit", () => {
    expect(subjectMaskFileError(null)).toMatch(/Choose a mask/);
    expect(subjectMaskFileError({ type: "image/gif", size: 10 })).toMatch(/PNG, JPEG, or WebP/);
    expect(subjectMaskFileError({ type: "image/png", size: SUBJECT_MASK_MAX_UPLOAD_BYTES + 1 })).toMatch(/32 MiB/);
  });
});

describe("subject mask report helpers", () => {
  it("indexes statuses by item id and versions the mask URL", () => {
    const byId = subjectMaskStatusByItemId({ items: [{ itemId: "a", maskPath: "training/datasets/d/masks/h.png" }] });
    expect(byId.get("a").maskPath).toBe("training/datasets/d/masks/h.png");
    expect(subjectMaskUrl("p", { maskPath: "training/datasets/d/masks/h.png", revision: "0123456789abcdef" })).toMatch(
      /\/api\/v1\/projects\/p\/files\/training\/datasets\/d\/masks\/h\.png\?v=0123456789abcdef$/,
    );
    expect(subjectMaskUrl("p", { hasMask: false })).toBe("");
  });

  it("versions the mask URL by content revision, not the 1-second updatedAt", () => {
    // Two uploads in the same second with different bytes: same path and updatedAt, new revision.
    const first = { maskPath: "training/datasets/d/masks/h.png", updatedAt: "2026-10-04T00:00:00Z", revision: "aaaaaaaaaaaaaaaa" };
    const second = { ...first, revision: "bbbbbbbbbbbbbbbb" };
    expect(subjectMaskUrl("p", second)).not.toBe(subjectMaskUrl("p", first));
  });
});
