import { describe, expect, it } from "vitest";

import {
  hasVisibleLocalFailureForView,
  isCurrentProjectRequest,
  reconcileSelectedAssetId,
} from "./appStateHelpers.js";

describe("hasVisibleLocalFailureForView", () => {
  const localJobIds = {
    image: ["image-local"],
    video: ["video-local"],
    audio: ["audio-local"],
    document: ["document-local"],
  };

  it("treats a locally launched Audio job as visible while Audio Studio is active", () => {
    expect(
      hasVisibleLocalFailureForView("Audio", localJobIds, {
        id: "audio-local",
        type: "audio_generate",
      }),
    ).toBe(true);
  });

  it("does not suppress the global notice for an Audio job outside Audio Studio", () => {
    expect(
      hasVisibleLocalFailureForView("Assets", localJobIds, {
        id: "audio-local",
        type: "audio_generate",
      }),
    ).toBe(false);
  });
});

describe("reconcileSelectedAssetId", () => {
  const items = [
    { id: "available", status: {} },
    { id: "trashed", status: { trashed: true } },
  ];

  it("retains a selection only while that id exists in the fetched project catalog", () => {
    expect(reconcileSelectedAssetId(items, "available")).toBe("available");
    expect(reconcileSelectedAssetId(items, "other-project")).toBe("available");
  });

  it("preserves the initial default selection and all-rejected fallback behavior", () => {
    expect(reconcileSelectedAssetId(items, null)).toBe("available");
    expect(
      reconcileSelectedAssetId(
        [{ id: "rejected", status: { rejected: true } }],
        null,
      ),
    ).toBe("rejected");
    expect(reconcileSelectedAssetId([], "gone")).toBeNull();
  });
});

describe("isCurrentProjectRequest", () => {
  it("rejects a project response after the active project is cleared", () => {
    expect(isCurrentProjectRequest(null, "project-a")).toBe(false);
  });

  it("accepts only the exact currently active project", () => {
    expect(isCurrentProjectRequest("project-a", "project-a")).toBe(true);
    expect(isCurrentProjectRequest("project-b", "project-a")).toBe(false);
  });
});
