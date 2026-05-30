import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Control the mocked API: GET /assets returns `poseAssets`, mutations resolve and are
// recorded in `apiCalls`.
const apiCalls = [];
let poseAssets = [];

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual, // keep API_BASE_URL / eventUrl / isAbortError for assetMedia etc.
    apiFetch: vi.fn(async (path, _token, options = {}) => {
      const method = options.method ?? "GET";
      apiCalls.push({ path, method });
      if (method === "GET" && path.includes("/assets")) {
        return poseAssets;
      }
      return {};
    }),
  };
});

import { AppContext } from "../context/AppContext.js";
import { PoseLibraryScreen } from "./PoseLibraryScreen.jsx";

function poseAsset(overrides = {}) {
  return {
    id: "asset_pose_1",
    projectId: "project_global_poses",
    type: "pose",
    displayName: "Arm Raised",
    tags: ["dynamic"],
    file: { path: "assets/poses/asset_pose_1.png", mimeType: "image/png", width: 768, height: 1280 },
    url: "/api/v1/projects/project_global_poses/files/assets/poses/asset_pose_1.png",
    status: { favorite: false, rating: 0, rejected: false, trashed: false },
    pose: { category: "dance", keypoints: [[0.5, 0.1]] },
    recipe: {},
    lineage: {},
    ...overrides,
  };
}

async function click(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

describe("PoseLibraryScreen", () => {
  let container;
  let root;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    apiCalls.length = 0;
    poseAssets = [];
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render() {
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ token: "test-token" }}>
          <PoseLibraryScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {}); // flush the refresh() effect
  }

  it("fetches reserved-project poses and groups them by category", async () => {
    poseAssets = [poseAsset()];
    await render();
    expect(apiCalls[0].path).toContain("/api/v1/projects/project_global_poses/assets");
    expect(container.textContent).toContain("Arm Raised");
    expect(container.textContent).toContain("dance");
  });

  it("shows an empty state when there are no saved poses", async () => {
    poseAssets = [];
    await render();
    expect(container.textContent).toContain("No saved poses yet");
  });

  it("discards a selected pose against the reserved project", async () => {
    poseAssets = [poseAsset()];
    await render();
    const tile = [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Arm Raised"));
    await click(tile);
    const discard = [...container.querySelectorAll("button")].find((b) => b.textContent.trim() === "Discard");
    expect(discard).toBeTruthy();
    await click(discard);
    expect(
      apiCalls.some((c) => c.method === "DELETE" && c.path === "/api/v1/projects/project_global_poses/assets/asset_pose_1"),
    ).toBe(true);
  });

  it("switches to the Create tab", async () => {
    poseAssets = [poseAsset()];
    await render();
    const createTab = container.querySelector("#pose-library-tab-create");
    await click(createTab);
    expect(container.querySelector("#pose-library-panel-poses").hidden).toBe(true);
    expect(container.querySelector("#pose-library-panel-create").hidden).toBe(false);
  });
});
