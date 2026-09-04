import fs from "node:fs";
import JSON5 from "json5";
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
  vectorDetailBudget,
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
    vector: {
      acceptsTextGuidance: false,
      providers: { mlx: { id: "mlx-starvector", available: true }, candle: { id: "candle-starvector", available: true } },
    },
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

  async function render(models, createVectorPromptWorkflow = vi.fn(async () => ({})), overrides = {}) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={{
          activeProject: { id: "p1", name: "Vectors" },
          assets: overrides.assets ?? [],
          jobs: [],
          models,
          macCapabilities: { platform: "macos" },
          macCapabilitiesAuthoritative: true,
          createVectorJob: overrides.createVectorJob ?? vi.fn(),
          createVectorPromptWorkflow,
          createModelDownloadJob: vi.fn(),
          setActiveView: vi.fn(),
          studioLaunch: overrides.studioLaunch,
        }}>
          <VectorStudio key={overrides.renderKey} />
        </AppContext.Provider>,
      );
    });
    return createVectorPromptWorkflow;
  }

  const source = { id: "source", projectId: "p1", type: "image", file: { mimeType: "image/png" }, status: {} };
  const sampling = { seed: 123, temperature: 0.8, topP: 0.7, topK: 5, repetitionPenalty: 1.2, repetitionContext: 40 };
  const budget = { maxNewTokens: 2345, maxSvgBytes: 150000, maxWallTimeMs: 67000 };
  async function submit() {
    await act(async () => container.querySelector("form").dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
  }

  it("submits every preset in both modes within the real catalog and wire DTO", async () => {
    const catalog = JSON5.parse(fs.readFileSync("../../config/manifests/builtin.models.jsonc", "utf8"));
    for (const manifest of catalog.models.filter((model) => model.type === "vector")) {
      const create = vi.fn(async () => ({}));
      const convert = vi.fn(async () => ({}));
      await render([rasterModel(), vectorModel({ ...manifest, installState: "installed", cacheState: "complete" })], create, {
        renderKey: manifest.id, assets: [source], createVectorJob: convert, studioLaunch: { view: "VectorStudio", assetId: source.id },
      });
      for (const mode of ["Convert Image", "Create from Prompt"]) {
        await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === mode).click());
        if (mode === "Create from Prompt") {
          await act(async () => {
            const input = container.querySelector("textarea");
            Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set.call(input, "a circle");
            input.dispatchEvent(new Event("input", { bubbles: true }));
          });
        }
        for (const preset of Object.keys(VECTOR_DETAIL_PRESETS)) {
          await act(async () => {
            const select = container.querySelector('[aria-label="Vector detail"]');
            select.value = preset;
            select.dispatchEvent(new Event("change", { bubbles: true }));
          });
          await submit();
          const dto = (mode === "Convert Image" ? convert : create).mock.lastCall[0].detailBudget;
          expect(Object.keys(dto).sort()).toEqual(["maxNewTokens", "maxSvgBytes", "maxWallTimeMs"].sort());
          for (const key of Object.keys(dto)) {
            expect(dto[key]).toBeGreaterThan(0);
            expect(dto[key]).toBeLessThanOrEqual(manifest.vector[key]);
          }
        }
      }
    }
  });

  it("replays exact conversion inputs after a composed recipe and preserves custom detail", async () => {
    const convert = vi.fn(async () => ({}));
    const models = [rasterModel(), vectorModel(), vectorModel({ id: "starvector_8b" })];
    await render(models, undefined, { studioLaunch: { view: "VectorStudio", recipe: { workflow: {
      kind: "create_from_prompt", rasterStage: { prompt: "circle", model: "flux_schnell" }, vectorStage: { model: "starvector_1b" },
    } } } });
    await render(models, undefined, { assets: [source], createVectorJob: convert, studioLaunch: {
      view: "VectorStudio", assetId: source.id,
      recipe: { mode: "image_to_svg", model: "starvector_8b", prompt: "", sampling, detailBudget: budget },
    } });
    await submit();
    expect(convert).toHaveBeenLastCalledWith({ mode: "image_to_svg", model: "starvector_8b", sourceAssetId: "source", prompt: "", sampling, detailBudget: budget });
    await act(async () => {
      const select = container.querySelector('[aria-label="Vector detail"]');
      select.value = "draft";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await submit();
    expect(convert.mock.lastCall[0]).toMatchObject({ sampling, detailBudget: vectorDetailBudget(VECTOR_DETAIL_PRESETS.draft, models[2]) });
  });

  it("replays both composed stages including seed dimensions sampling and exact revisions", async () => {
    const create = vi.fn(async () => ({}));
    await render([rasterModel(), vectorModel(), vectorModel({ id: "starvector_8b" })], create, { studioLaunch: {
      view: "VectorStudio", recipe: { workflow: {
        kind: "create_from_prompt",
        rasterStage: { model: "flux_schnell", revision: RASTER_REVISION, prompt: "circle", negativePrompt: "noise", seed: 456, width: 768, height: 512 },
        vectorStage: { model: "starvector_8b", revision: VECTOR_REVISION, sampling, detailBudget: budget },
      } },
    } });
    await submit();
    expect(create).toHaveBeenLastCalledWith({ prompt: "circle", negativePrompt: "noise", rasterModel: "flux_schnell", vectorModel: "starvector_8b", seed: 456, width: 768, height: 512, sampling, detailBudget: budget, expectedRasterRevision: RASTER_REVISION, expectedVectorRevision: VECTOR_REVISION });
  });

  it("does not substitute an unavailable recorded model or clamp recorded budgets", async () => {
    const convert = vi.fn(async () => ({}));
    for (const recipe of [
      { model: "missing", detailBudget: budget },
      { model: "starvector_1b", detailBudget: { ...budget, maxNewTokens: 5000 } },
    ]) {
      await render([vectorModel(), vectorModel({ id: "starvector_8b" })], undefined, { assets: [source], createVectorJob: convert, studioLaunch: {
        view: "VectorStudio", assetId: source.id, recipe: { mode: "image_to_svg", sampling, ...recipe },
      } });
      expect(container.querySelector('[role="alert"]').textContent).toContain("This recipe cannot run");
      if (container.querySelector("form")) await submit();
    }
    expect(convert).not.toHaveBeenCalled();
  });

  it("does not expose the workflow tab until both stages are eligible", async () => {
    await render([vectorModel()]);
    expect(container.textContent).not.toContain("Create from Prompt");
  });

  it("keeps the installed 8B tier out of conversion until its terminal candidate exists", async () => {
    await render([
      vectorModel(),
      vectorModel({
        id: "starvector_8b",
        name: "StarVector-8B",
        vector: { providers: {
          mlx: { id: "mlx-starvector-8b", available: false, reason: "pending_terminal_candidate" },
          candle: { id: "candle-starvector-8b", available: false, reason: "pending_terminal_candidate" },
        } },
      }),
    ]);
    expect(container.querySelector('[aria-label="Conversion model"]')).toBeNull();
    expect(container.textContent).not.toContain("Text to SVG");
  });

  it("discloses the typed 8B terminal-candidate refusal when it is the only installed tier", async () => {
    await render([
      vectorModel({
        id: "starvector_8b",
        name: "StarVector-8B",
        vector: { providers: {
          mlx: { id: "mlx-starvector-8b", available: false, reason: "pending_terminal_candidate" },
          candle: { id: "candle-starvector-8b", available: false, reason: "pending_terminal_candidate" },
        } },
      }),
    ]);
    expect(container.textContent).toContain("dispatch stays disabled until its permanent-pin terminal candidate is accepted");
    expect(container.querySelector('[aria-label="Optional vector guidance"]')).toBeNull();
  });

  it("never collects or sends text guidance for direct 1B image-only requests", async () => {
    const createVectorJob = vi.fn(async () => ({}));
    const source = {
      id: "source-png",
      projectId: "p1",
      type: "image",
      file: { mimeType: "image/png" },
      status: {},
    };
    await render(
      [vectorModel()],
      undefined,
      {
        assets: [source],
        createVectorJob,
        studioLaunch: { view: "VectorStudio", assetId: source.id },
      },
    );
    expect(container.querySelector('[aria-label="Optional vector guidance"]')).toBeNull();

    const form = container.querySelector("form");
    await act(async () => form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
    expect(createVectorJob).toHaveBeenLastCalledWith(expect.objectContaining({
      mode: "image_to_svg",
      model: "starvector_1b",
      prompt: "",
    }));
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
      detailBudget: vectorDetailBudget(VECTOR_DETAIL_PRESETS.standard, vectorModel()),
    }));
  });
});
