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
      maxNewTokens: 7933,
      maxSvgBytes: 262144,
      maxWallTimeMs: 300000,
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
    expect(VECTOR_DETAIL_PRESETS.draft.maxWallTimeMs).toBe(60000);
    expect(VECTOR_DETAIL_PRESETS.standard.maxWallTimeMs).toBe(90000);
    expect(VECTOR_DETAIL_PRESETS.detailed.maxWallTimeMs).toBe(300000);
  });

  it("uses each selected model's declared context-derived detailed token cap", () => {
    expect(
      vectorDetailBudget(
        VECTOR_DETAIL_PRESETS.detailed,
        vectorModel({ vector: { maxNewTokens: 7933, maxSvgBytes: 262144, maxWallTimeMs: 300000 } }),
      ),
    ).toMatchObject({ maxNewTokens: 7933 });
    expect(
      vectorDetailBudget(
        VECTOR_DETAIL_PRESETS.detailed,
        vectorModel({
          id: "starvector_8b",
          vector: { maxNewTokens: 15422, maxSvgBytes: 262144, maxWallTimeMs: 300000 },
        }),
      ),
    ).toMatchObject({ maxNewTokens: 15422 });
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
          activeProject: overrides.activeProject ?? { id: "p1", name: "Vectors" },
          assets: overrides.assets ?? [],
          assetsLoading: overrides.assetsLoading ?? false,
          jobs: overrides.jobs ?? [],
          models,
          macCapabilities: { platform: "macos" },
          macCapabilitiesAuthoritative: true,
          createVectorJob: overrides.createVectorJob ?? vi.fn(),
          createVectorPromptWorkflow,
          createModelDownloadJob: vi.fn(),
          setActiveView: overrides.setActiveView ?? vi.fn(),
          setSelectedAssetId: overrides.setSelectedAssetId ?? vi.fn(),
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
      { model: "starvector_1b", detailBudget: { ...budget, maxNewTokens: 8000 } },
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
      detailBudget: vectorDetailBudget(VECTOR_DETAIL_PRESETS.detailed, vectorModel()),
    }));
  });

  it("defaults both fresh modes to the selected model's full Detailed budget", async () => {
    const convert = vi.fn(async () => ({}));
    const create = vi.fn(async () => ({}));
    await render([rasterModel(), vectorModel()], create, { assets: [source], createVectorJob: convert, studioLaunch: { view: "VectorStudio", assetId: source.id } });
    expect(container.querySelector('[aria-label="Vector detail"]').value).toBe("detailed");
    expect(container.textContent).toContain("Selected vector budget: 7,933 tokens · 300s");
    await submit();
    expect(convert).toHaveBeenLastCalledWith(expect.objectContaining({ detailBudget: vectorDetailBudget(VECTOR_DETAIL_PRESETS.detailed, vectorModel()) }));
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "Create from Prompt").click());
    const prompt = container.querySelector('[aria-label="Vector prompt"]');
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set.call(prompt, "a geometric fox");
      prompt.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await submit();
    expect(create).toHaveBeenLastCalledWith(expect.objectContaining({ detailBudget: vectorDetailBudget(VECTOR_DETAIL_PRESETS.detailed, vectorModel()) }));
  });

  it("retains only this project's terminal outcomes and opens a published safe vector result in Assets", async () => {
    const setActiveView = vi.fn();
    const setSelectedAssetId = vi.fn();
    const result = {
      id: "vector-result",
      projectId: "p1",
      type: "vector",
      displayName: "geometric-fox.svg",
      file: { path: "assets/vector-result/vector.svg", mimeType: "image/svg+xml" },
      preview: { path: "assets/vector-result/preview.png", mimeType: "image/png" },
      status: {},
    };
    const completed = { id: "job-p1", projectId: "p1", type: "vector_generate", status: "completed", result: { assetWrites: [{ id: result.id }] } };
    const foreign = { id: "job-p2", projectId: "p2", type: "vector_generate", status: "completed", result: { assetWrites: [{ id: result.id }] } };
    await render([vectorModel()], undefined, { assets: [result], jobs: [foreign, completed], setActiveView, setSelectedAssetId });
    expect(container.textContent).toContain("Saved geometric-fox.svg");
    expect(container.querySelector("img").src).toContain("preview.png");
    expect(container.querySelector('a[download]').href).toContain("vector.svg");
    await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === "View in Assets").click());
    expect(setSelectedAssetId).toHaveBeenCalledWith(result.id);
    expect(setActiveView).toHaveBeenCalledWith("Library");
  });

  it("does not mark incomplete, failed, or cancelled terminal jobs as saved results", async () => {
    const missingAsset = { id: "job-saving", projectId: "p1", type: "vector_generate", status: "completed", result: { assetWrites: [{ id: "not-yet-published" }] } };
    await render([vectorModel()], undefined, { jobs: [missingAsset], assetsLoading: true });
    expect(container.textContent).toContain("Loading the saved SVG result");
    expect(container.textContent).not.toContain("View in Assets");
    await render([vectorModel()], undefined, { jobs: [missingAsset], assetsLoading: false });
    expect(container.textContent).toContain("SVG result unavailable");
    expect(container.textContent).not.toContain("Loading the saved SVG result");
    expect(container.querySelector('a[download]')).toBeNull();
  });

  it("shows a removed completed result as unavailable rather than saving forever", async () => {
    const result = { id: "purged-vector", projectId: "p1", type: "vector", displayName: "removed.svg" };
    const completed = { id: "job-purged", projectId: "p1", type: "vector_generate", status: "completed", result: { assets: [result] } };
    await render([vectorModel()], undefined, { assets: [result], jobs: [completed] });
    expect(container.textContent).toContain("Saved removed.svg");
    await render([vectorModel()], undefined, { assets: [], jobs: [completed] });
    expect(container.textContent).toContain("SVG result unavailable");
    expect(container.textContent).not.toContain("Saved removed.svg");
    expect(container.querySelector('a[download]')).toBeNull();
  });

  it("keeps incomplete recorded budgets recoverable without crashing or rewriting them", async () => {
    const incompleteBudget = { maxSvgBytes: 262144, maxWallTimeMs: 300000 };
    const convert = vi.fn();
    await render([vectorModel()], undefined, {
      assets: [source], createVectorJob: convert,
      studioLaunch: { view: "VectorStudio", assetId: source.id, recipe: { mode: "image_to_svg", detailBudget: incompleteBudget } },
    });
    expect(container.textContent).toContain("Invalid recorded budget");
    expect(container.querySelector('[role="alert"]').textContent).toContain("This recipe cannot run");
    await submit();
    expect(convert).not.toHaveBeenCalled();
  });

  it("keeps failed and cancelled project jobs visible without treating either as a result", async () => {
    const failed = { id: "job-failed", projectId: "p1", type: "vector_generate", status: "failed", error: "SVG sanitizer rejected the output" };
    const cancelled = { id: "job-cancelled", projectId: "p1", type: "vector_generate", status: "canceled" };
    await render([vectorModel()], undefined, { jobs: [failed] });
    expect(container.textContent).toContain("Failed");
    expect(container.textContent).toContain("SVG sanitizer rejected the output");
    expect(container.textContent).not.toContain("Saved SVG result");
    await render([vectorModel()], undefined, { jobs: [cancelled] });
    expect(container.textContent).toContain("Cancelled");
    expect(container.textContent).not.toContain("Saved SVG result");
  });

  it("shows rejected submissions locally without changing exact replay inputs", async () => {
    const convert = vi.fn(async () => { throw new Error("provider refused this request"); });
    await render([vectorModel({ id: "starvector_8b", vector: { ...vectorModel().vector, maxNewTokens: 9000 } })], undefined, {
      assets: [source],
      createVectorJob: convert,
      studioLaunch: { view: "VectorStudio", assetId: source.id, recipe: { mode: "image_to_svg", model: "starvector_8b", sampling, detailBudget: budget } },
    });
    await submit();
    expect(container.querySelector('[role="alert"]').textContent).toContain("provider refused this request");
    await submit();
    expect(convert).toHaveBeenCalledTimes(2);
    expect(convert).toHaveBeenLastCalledWith(expect.objectContaining({ sampling, detailBudget: budget }));
  });
});
