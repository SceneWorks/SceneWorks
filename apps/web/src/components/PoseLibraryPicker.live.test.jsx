import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getMediaTicket, setMediaTicket } from "../api.js";
import { poseAssetToRecord } from "../poseLibrary.js";
import { PoseLibraryPicker } from "./PoseLibraryPicker.jsx";

describe("PoseLibraryPicker with the real pose-library hook", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    setMediaTicket("");
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("does not reload or loop when a parent rerenders without extra poses", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        poses: [
          {
            id: "standing-1",
            label: "Standing 1",
            category: "standing",
            keypoints: [[0.5, 0.5]],
            preview: "poses/standing-1.png",
          },
        ],
      }),
    }));
    vi.stubGlobal("fetch", fetchMock);

    const renderPicker = async (tick) => {
      await act(async () => {
        root.render(
          <div data-tick={tick}>
            <PoseLibraryPicker onToggle={vi.fn()} selectedIds={[]} />
          </div>,
        );
      });
      await act(async () => {});
    };

    await renderPicker(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(container.querySelector('[aria-label="Select pose Standing 1"]')).not.toBeNull();

    await renderPicker(2);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(container.querySelector("[data-tick='2']")).not.toBeNull();
  });
  it("uses the current ticket for a newly opened cached user-pose category", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ poses: [] }) })));
    setMediaTicket("old-ticket");
    const assets = ["First", "Later"].map((label) => ({
      id: label, type: "pose", displayName: label,
      projectId: "project_global_poses",
      file: { path: `assets/${label}.png`, mimeType: "image/png" },
      pose: { category: label.toLowerCase(), keypoints: [] },
    }));
    const loadUserPoses = vi.fn(async () => assets.map(poseAssetToRecord));
    let oldExpired = false;
    const mediaServer = vi.fn((url) => ({
      status: new URL(url).searchParams.get("ticket") === "new-ticket" || !oldExpired ? 200 : 401,
    }));
    const originalSetAttribute = HTMLImageElement.prototype.setAttribute;
    const setAttribute = vi.spyOn(HTMLImageElement.prototype, "setAttribute").mockImplementation(function (name, value) {
      if (name === "src" && String(value).includes("/api/v1/projects/")) mediaServer(value);
      return originalSetAttribute.call(this, name, value);
    });
    await act(async () => root.render(<PoseLibraryPicker categoryFilter preferredCategory="first" loadUserPoses={loadUserPoses} onToggle={vi.fn()} />));
    await act(async () => {});
    expect(container.querySelector('img[alt="First"]').src).toContain("ticket=old-ticket");
    expect(loadUserPoses).toHaveBeenCalledTimes(1);
    oldExpired = true;
    await act(async () => setMediaTicket("new-ticket"));
    expect(getMediaTicket()).toBe("new-ticket");
    setAttribute.mockClear();
    mediaServer.mockClear();
    const category = [...container.querySelectorAll(".pose-chip")].find((button) => button.textContent.includes("Later"));
    await act(async () => category.click());
    const img = container.querySelector('img[alt="Later"]');
    expect(img.src).toContain("ticket=new-ticket");
    expect(img.src).not.toContain("old-ticket");
    const requests = setAttribute.mock.calls.filter(([name]) => name === "src").map(([, value]) => value);
    expect(requests).toHaveLength(1);
    expect(requests[0]).toContain("ticket=new-ticket");
    expect(mediaServer).toHaveBeenCalledTimes(1);
    expect(mediaServer.mock.results[0].value.status).toBe(200);
    expect(mediaServer("http://localhost:8000/api/v1/projects/p/files/a.png?ticket=old-ticket").status).toBe(401);
    expect(loadUserPoses).toHaveBeenCalledTimes(1);
  });

});
