import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// konva's node build pulls in the native `canvas` package (not installed, and not
// usable in jsdom). The empty-state paths under test never mount the <Stage>, so
// stub react-konva to keep konva out of the import graph — mirroring how App.jsx
// lazy-loads the editor to keep konva off the test/initial path.
vi.mock("react-konva", async () => {
  const React = await import("react");
  const passthrough = (name) => ({ children }) => React.createElement("div", { "data-konva": name }, children);
  return { Stage: passthrough("stage"), Layer: passthrough("layer"), Image: () => null, Rect: () => null };
});

import { AppContext } from "../context/AppContext.js";
import { ImageEditor, cropRatioForKey, centeredCropRect } from "./ImageEditor.jsx";

// These tests cover the non-canvas surface of the editor (empty state, the inert
// tool scaffold, and the load affordances). The Konva <Stage> only mounts once a
// working image is present, which needs a real canvas — out of reach for jsdom —
// so canvas behaviour (zoom/pan/fit) is verified in the browser, not here. Simply
// mounting also asserts that importing react-konva/konva doesn't break jsdom.
function baseContext(overrides = {}) {
  return {
    activeProject: null,
    assets: [],
    setPreviewAsset: vi.fn(),
    ...overrides,
  };
}

const toolButtons = (container) => [...container.querySelectorAll(".image-editor-tool")];
const actionButton = (container, label) =>
  [...container.querySelectorAll(".image-editor-actions button")].find((b) => b.textContent.trim() === label);

describe("ImageEditor scaffold", () => {
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
    vi.clearAllMocks();
  });

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageEditor />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("renders the empty state and inert tool scaffold without a working image", async () => {
    await render(baseContext());

    expect(container.textContent).toContain("Open an image to start editing");
    // No working image → no Konva stage / view controls yet.
    expect(container.querySelector(".image-editor-viewbar")).toBeNull();

    const tools = toolButtons(container);
    expect(tools.map((b) => b.textContent.trim())).toEqual(["Move", "Crop", "Upscale", "AI Edit", "Detail", "Color"]);
    // Move is the active default. Without a loaded image Crop is gated off, and the
    // remaining tools are inert placeholders for later slices — so all but Move are
    // disabled here.
    expect(tools[0].disabled).toBe(false);
    expect(tools.slice(1).every((b) => b.disabled)).toBe(true);
  });

  it("gates 'Open from project' on an active project but always offers upload", async () => {
    await render(baseContext());
    expect(actionButton(container, "Open from project").disabled).toBe(true);
    expect(actionButton(container, "Upload image")).toBeTruthy();

    await render(baseContext({ activeProject: { id: "project_1", name: "My Project" } }));
    expect(actionButton(container, "Open from project").disabled).toBe(false);
  });
});

describe("crop geometry", () => {
  it("resolves ratio keys, transposing only non-square ratios on rotate", () => {
    expect(cropRatioForKey("free", false)).toBeNull();
    expect(cropRatioForKey("1:1", false)).toBe(1);
    expect(cropRatioForKey("1:1", true)).toBe(1); // square is unaffected by rotate
    expect(cropRatioForKey("16:9", false)).toBeCloseTo(16 / 9);
    expect(cropRatioForKey("16:9", true)).toBeCloseTo(9 / 16);
    expect(cropRatioForKey("3:4", true)).toBeCloseTo(4 / 3);
  });

  it("centers the largest rect of the ratio that fits in the image", () => {
    // Square in a landscape image → limited by height, centered horizontally.
    expect(centeredCropRect(1000, 500, 1)).toEqual({ x: 250, y: 0, width: 500, height: 500 });
    // 16:9 in a 1000×500 image → limited by height (562.5 > 500), centered.
    const wide = centeredCropRect(1000, 500, 16 / 9);
    expect(wide.height).toBe(500);
    expect(wide.width).toBeCloseTo((500 * 16) / 9);
    expect(wide.x).toBeCloseTo((1000 - (500 * 16) / 9) / 2);
    expect(wide.y).toBe(0);
    // Freeform → centered 80% box.
    expect(centeredCropRect(800, 600, null)).toEqual({ x: 80, y: 60, width: 640, height: 480 });
  });
});
