import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import {
  VECTOR_DETAIL_PRESETS,
  VectorStudio,
  authoritativeModelRevision,
  promptVectorModelAvailability,
  rasterPromptModelAvailability,
  vectorSourceAssets,
} from "./VectorStudio.jsx";

const RASTER_REVISION = "1111111111111111111111111111111111111111";
const VECTOR_REVISION = "2222222222222222222222222222222222222222";

function rasterModel(overrides = {}) {
  return {
    id: "flux_schnell",
    name: "FLUX Schnell",
    type: "image",
    capabilities: ["text_to_image"],
    installState: "installed",
    cacheState: "complete",
    usable: true,
    macSupport: { supported: true },
    candleSupport: { supported: true },
    downloads: [{ revision: RASTER_REVISION }],
    ...overrides,
  };
}

function vectorModel(overrides = {}) {
  return {
    id: "starvector_1b",
    name: "StarVector-1B",
    type: "vector",
    capabilities: ["image_to_svg"],
    installState: "installed",
    cacheState: "complete",
    vector: { providers: { mlx: { id: "mlx-starvector", available: true }, candle: { id: "candle-starvector", available: true } } },
    downloads: [{ revision: VECTOR_REVISION }],
    ...overrides,
  };
}

describe("Vector Studio source boundary", () => {
  it("admits only active project raster assets and keeps SVG out of conditioning", () => {
    const assets = [
      { id: "png", projectId: "p1", type: "image", file: { mimeType: "image/png" }, status: {} },
      { id: "svg", projectId: "p1", type: "vector", file: { mimeType: "image/svg+xml" }, preview: { path: "preview.png" }, status: {} },
      { id: "other", projectId: "p2", type: "image", file: { mimeType: "image/png" }, status: {} },
      { id: "trashed", projectId: "p1", type: "image", file: { mimeType: "image/png" }, status: { trashed: true } },
    ];
    expect(vectorSourceAssets(assets, "p1").map((asset) => asset.id)).toEqual(["png"]);
  });

  it("keeps every selectable detail payload bounded", () => {
    for (const preset of Object.values(VECTOR_DETAIL_PRESETS)) {
      expect(preset.maxNewTokens).toBeGreaterThan(0);
      expect(preset.maxSvgBytes).toBeGreaterThan(0);
      expect(preset.maxWallTimeMs).toBeGreaterThan(0);
    }
  });
});

describe("Create from Prompt eligibility", () => {
  it("requires installed claimable stages with one immutable revision", () => {
    const caps = { platform: "macos" };
    expect(rasterPromptModelAvailability(rasterModel(), caps)).toMatchObject({ available: true, revision: RASTER_REVISION });
    expect(promptVectorModelAvailability(vectorModel(), caps)).toMatchObject({ available: true, revision: VECTOR_REVISION });
    expect(rasterPromptModelAvailability(rasterModel({ installState: "missing" }), caps).available).toBe(false);
    expect(rasterPromptModelAvailability(rasterModel({ macSupport: { supported: false } }), caps).reason).toBe("backend_unclaimable");
    expect(authoritativeModelRevision(rasterModel({ downloads: [] }))).toBeNull();
    expect(authoritativeModelRevision(rasterModel({ downloads: [{ revision: "main" }] }))).toBeNull();
    expect(authoritativeModelRevision(rasterModel({ downloads: [{ revision: RASTER_REVISION }, { revision: "main" }] }))).toBeNull();
    expect(authoritativeModelRevision(rasterModel({ downloads: [{ revision: RASTER_REVISION }, { revision: VECTOR_REVISION }] }))).toBeNull();
    expect(promptVectorModelAvailability(vectorModel({ capabilities: ["text_to_svg"] }), caps).available).toBe(false);
  });
});

describe("Create from Prompt disclosure", () => {
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
    delete global.IS_REACT_ACT_ENVIRONMENT;
  });

  async function render(models, createVectorPromptWorkflow = vi.fn(async () => ({}))) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={{
          activeProject: { id: "p1", name: "Vectors" },
          assets: [],
          jobs: [],
          models,
          macCapabilities: { platform: "macos" },
          macCapabilitiesAuthoritative: true,
          createVectorJob: vi.fn(),
          createVectorPromptWorkflow,
          createModelDownloadJob: vi.fn(),
          setActiveView: vi.fn(),
        }}>
          <VectorStudio />
        </AppContext.Provider>,
      );
    });
    return createVectorPromptWorkflow;
  }

  it("does not expose the workflow tab until both stages are eligible", async () => {
    await render([vectorModel()]);
    expect(container.textContent).not.toContain("Create from Prompt");
  });

  it("fails closed while current-backend capability facts are not authoritative", async () => {
    await act(async () => {
      root.render(
        <AppContext.Provider value={{
          activeProject: { id: "p1", name: "Vectors" },
          assets: [],
          jobs: [],
          models: [rasterModel(), vectorModel()],
          macCapabilities: { platform: "macos" },
          macCapabilitiesAuthoritative: false,
          createVectorJob: vi.fn(),
          createVectorPromptWorkflow: vi.fn(),
          createModelDownloadJob: vi.fn(),
          setActiveView: vi.fn(),
        }}>
          <VectorStudio />
        </AppContext.Provider>,
      );
    });
    expect(container.textContent).not.toContain("Create from Prompt");
  });

  it("names both exact stages, disclaims direct text mode, and submits the composition", async () => {
    const create = await render([rasterModel(), vectorModel()]);
    const tab = [...container.querySelectorAll("button")].find((button) => button.textContent === "Create from Prompt");
    expect(tab).toBeTruthy();
    await act(async () => tab.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    const disclosure = container.querySelector('[role="note"]');
    expect(disclosure.textContent).toContain("not direct text-to-SVG");
    expect(disclosure.textContent).toContain(RASTER_REVISION);
    expect(disclosure.textContent).toContain(VECTOR_REVISION);
    const prompt = container.querySelector('[aria-label="Vector prompt"]');
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
      setter.call(prompt, "a geometric fox");
      prompt.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const form = prompt.closest("form");
    await act(async () => form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
    expect(create).toHaveBeenCalledWith(expect.objectContaining({
      prompt: "a geometric fox",
      rasterModel: "flux_schnell",
      vectorModel: "starvector_1b",
      detailBudget: VECTOR_DETAIL_PRESETS.standard,
    }));
  });
});
