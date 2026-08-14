import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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
});
