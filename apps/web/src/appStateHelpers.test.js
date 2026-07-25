import { describe, expect, it } from "vitest";

import appSource from "./App.jsx?raw";
import {
  applySuccessfulProjectRefresh,
  hasVisibleLocalFailureForView,
  isCurrentProjectRequest,
  reconcileActiveProject,
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

  it("rejects the previous project during the layout-to-passive switch window", () => {
    // Model the App commit window directly: refreshAssets already closes over B,
    // while activeProjectRef still says A until its passive effect runs. The
    // pre-fetch decision must use the render-current B id.
    const renderCurrentProjectId = "project-b";
    const passiveRefProjectId = "project-a";
    const sseProjectId = "project-a";

    expect(passiveRefProjectId).toBe(sseProjectId);
    expect(isCurrentProjectRequest(renderCurrentProjectId, sseProjectId)).toBe(false);

    const refreshStart = appSource.slice(
      appSource.indexOf("async function refreshAssets"),
      appSource.indexOf("try {", appSource.indexOf("async function refreshAssets")),
    );
    expect(refreshStart).toContain(
      "isCurrentProjectRequest(activeProject?.id ?? null, projectId)",
    );
    expect(refreshStart).not.toContain("activeProjectRef.current");
  });
});

describe("reconcileActiveProject", () => {
  const refreshed = [
    { id: "project-a", name: "A refreshed", updatedAt: "new" },
    { id: "project-b", name: "B" },
  ];

  it("replaces a selected project's stale snapshot with the refreshed object", () => {
    expect(reconcileActiveProject(refreshed, { id: "project-a", name: "A stale" })).toBe(
      refreshed[0],
    );
  });

  it("selects the first current project when the old selection disappeared", () => {
    expect(reconcileActiveProject(refreshed, { id: "deleted" })).toBe(refreshed[0]);
    expect(reconcileActiveProject([], { id: "deleted" })).toBeNull();
  });
});

describe("applySuccessfulProjectRefresh", () => {
  it("retains the current project list and selection when the projects request fails", () => {
    let projectWrites = 0;
    let activeProjectWrites = 0;
    const applied = applySuccessfulProjectRefresh(
      { value: [], error: "Projects: offline" },
      () => {
        projectWrites += 1;
      },
      () => {
        activeProjectWrites += 1;
      },
    );

    expect(applied).toBe(false);
    expect(projectWrites).toBe(0);
    expect(activeProjectWrites).toBe(0);
  });

  it("updates both list and selected snapshot after a successful projects request", () => {
    const refreshed = [{ id: "project-a", name: "A refreshed" }];
    let writtenProjects = null;
    let activeUpdater = null;
    const applied = applySuccessfulProjectRefresh(
      { value: refreshed, error: "" },
      (projects) => {
        writtenProjects = projects;
      },
      (updater) => {
        activeUpdater = updater;
      },
    );

    expect(applied).toBe(true);
    expect(writtenProjects).toBe(refreshed);
    expect(activeUpdater({ id: "project-a", name: "A stale" })).toBe(refreshed[0]);
  });
});
