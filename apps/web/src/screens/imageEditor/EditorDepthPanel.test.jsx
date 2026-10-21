import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, apiFetch: vi.fn() };
});

import { apiFetch } from "../../api.js";
import { AppContext } from "../../context/AppContext.js";
import { EditorDepthPanel } from "./EditorDepthPanel.jsx";

const working = { width: 4000, height: 3000, source: { kind: "asset", assetId: "asset_src", name: "harbor.jpg" } };
const baseContext = {
  activeProject: { id: "p1", name: "Proj" },
  assets: [],
  jobs: [],
  jobAction: vi.fn(),
  models: [{ id: "iris_3b_depth", installState: "installed" }],
  macCapabilities: { features: { imageDepth: { supported: true } } },
  requestedGpu: "auto",
  setPreviewAsset: vi.fn(),
  token: "",
};

describe("Image Editor depth tool (sc-25682)", () => {
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

  const render = (context, props = {}) =>
    act(() =>
      root.render(
        <AppContext.Provider value={{ ...baseContext, ...context }}>
          <EditorDepthPanel dirty={false} working={working} {...props} />
        </AppContext.Provider>,
      ),
    );
  const button = (label) => [...container.querySelectorAll("button")].find((b) => b.textContent.trim() === label);

  // Round 2 (4): the model default follows the catalog until the user picks — a catalog that loads
  // after mount (or a trained depth model installed later) is honoured.
  it("re-derives the default depth model while the selection is untouched", async () => {
    apiFetch.mockResolvedValue({ id: "job-depth" });
    await render({ models: [] });
    const trained = { id: "my_depth", name: "My depth", installState: "installed", trainedCheckpoint: { task: "depth" } };
    await render({ models: [{ id: "iris_3b_depth", installState: "missing" }, trained] });
    expect(button("Estimate depth").disabled).toBe(false);
    await act(async () => button("Estimate depth").click());
    expect(JSON.parse(apiFetch.mock.calls[0][2].body).irisModel).toBe("my_depth");
  });

  it("estimates from the saved source with the chosen resolution, shows progress, cancels, opens the result", async () => {
    apiFetch.mockResolvedValue({ id: "job-depth" });
    await render({});
    expect(container.textContent).toContain("Model input 1024×768");
    await act(async () => button("Estimate depth").click());
    expect(apiFetch).toHaveBeenCalledTimes(1);
    const [path, , options] = apiFetch.mock.calls[0];
    expect(path).toBe("/api/v1/image/depth/jobs");
    expect(JSON.parse(options.body)).toEqual({
      projectId: "p1",
      projectName: "Proj",
      sourceAssetId: "asset_src",
      resolution: { mode: "capped", maxSide: 1024 },
      requestedGpu: "auto",
      displayName: "harbor.jpg",
    });

    const running = { id: "job-depth", status: "running", progress: 0.3, message: "Estimating depth at 1024x768." };
    await render({ jobs: [running] });
    expect(container.textContent).toContain("Estimating depth at 1024x768. (30%)");
    expect(button("Estimate depth").disabled).toBe(true);
    await act(async () => button("Cancel depth estimation").click());
    expect(baseContext.jobAction).toHaveBeenCalledWith(running, "cancel");

    await render({ jobs: [{ id: "job-depth", status: "canceled" }] });
    expect(container.textContent).toContain("no depth map was saved");

    const result = { id: "asset_depth", type: "depth" };
    await render({ jobs: [{ id: "job-depth", status: "completed", result: { assets: [result] } }] });
    await act(async () => button("Open depth map").click());
    expect(baseContext.setPreviewAsset).toHaveBeenCalledWith(result);
  });

  it("refuses an unsaved canvas, a missing install, and states why", async () => {
    await render({}, { dirty: true });
    expect(container.textContent).toContain("Save your edits first");
    expect(button("Estimate depth").disabled).toBe(true);
    await render({ models: [{ id: "iris_3b_depth", installState: "missing" }] });
    expect(container.textContent).toContain("Install Iris 3B Depth in Models");
    expect(button("Estimate depth").disabled).toBe(true);
    await render({}, { working: { ...working, source: { kind: "upload", name: "x.png" } } });
    expect(container.textContent).toContain("Save this image to the Library first");
  });

  it("lists earlier depth maps of this image and reopens their settings", async () => {
    const earlier = {
      id: "asset_old",
      type: "depth",
      projectId: "p1",
      createdAt: "2026-10-10T00:00:00Z",
      lineage: { sourceAssetId: "asset_src" },
      recipe: { rawAdapterSettings: { depth: { resolution: { mode: "native" } } } },
      preview: { path: "assets/depth/x/p.png" },
    };
    await render({ assets: [earlier, { ...earlier, id: "other", lineage: { sourceAssetId: "elsewhere" } }] });
    expect(container.querySelectorAll(".depth-results li")).toHaveLength(1);
    expect(container.textContent).toContain("Native");
    expect(container.querySelector('input[aria-label="Depth max side"]')).not.toBeNull();
    await act(async () => button("Use these settings").click());
    expect(container.querySelector('input[aria-label="Depth max side"]')).toBeNull();
    apiFetch.mockResolvedValue({ id: "job-2" });
    await act(async () => button("Estimate depth").click());
    expect(JSON.parse(apiFetch.mock.calls[0][2].body).resolution).toEqual({ mode: "native" });
  });
});
