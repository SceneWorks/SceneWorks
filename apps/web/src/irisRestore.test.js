import { describe, expect, it } from "vitest";
import {
  IRIS_RESTORE_DEFAULTS,
  irisControlsFromRecord,
  irisRestoreControls,
  irisRestoreInstalled,
  isIrisEngine,
  normalizeIrisScale,
  restorePlanBodyForAssets,
  restorePlanBodyForRenders,
  restorePlanBodyForSize,
  restorePlanItemText,
  summarizeRestorePlans,
  trainedTaskModels,
} from "./irisRestore.js";
import { buildUpscaleJobBody, upscaleEngineIsRestoration } from "./imageJobs.js";
import { buildBatchJob } from "./batchOps.js";

const project = { id: "p1", name: "Project" };

describe("Iris restoration controls (sc-25683)", () => {
  it("defaults to the model's 4×, budgeted input, colour fix on", () => {
    expect(IRIS_RESTORE_DEFAULTS).toEqual({ scale: 4, inputSizing: "budgeted", colorFix: true });
    expect(irisRestoreControls({})).toEqual(IRIS_RESTORE_DEFAULTS);
  });

  it("accepts any positive scale, including 1× and fractions, and never coerces an invalid one", () => {
    expect(normalizeIrisScale(1)).toBe(1);
    expect(normalizeIrisScale("1.3")).toBe(1.3);
    expect(normalizeIrisScale(0)).toBeNull();
    expect(normalizeIrisScale(-2)).toBeNull();
    expect(normalizeIrisScale("abc")).toBeNull();
    expect(irisRestoreControls({ scale: 2.5, inputSizing: "original", colorFix: false })).toEqual({
      scale: 2.5,
      inputSizing: "original",
      colorFix: false,
    });
  });

  it("offers the engine only when its own task entry is fully installed", () => {
    expect(irisRestoreInstalled([{ id: "iris_3b_restore", installState: "installed" }])).toBe(true);
    expect(irisRestoreInstalled([{ id: "iris_3b_restore", installState: "incomplete" }])).toBe(false);
    // The generation model is a different task.
    expect(irisRestoreInstalled([{ id: "iris_3b", installState: "installed" }])).toBe(false);
    expect(irisRestoreInstalled(undefined)).toBe(false);
  });

  it("builds plan request bodies for a size and for assets", () => {
    expect(restorePlanBodyForSize({ width: 512, height: 384, scale: 1 })).toEqual({
      scale: 1,
      inputSizing: "budgeted",
      colorFix: true,
      width: 512,
      height: 384,
    });
    expect(restorePlanBodyForAssets({ projectId: "p1", assetIds: ["a", "b"], inputSizing: "original" })).toEqual({
      scale: 4,
      inputSizing: "original",
      colorFix: true,
      projectId: "p1",
      assetIds: ["a", "b"],
    });
  });

  it("shows the planner's honest summary and blocks on any refused item", () => {
    const response = {
      items: [
        { assetId: "a", plan: { summary: "budgeted: 683x512 input → 2732x2048 (4× of the budgeted input; source 4032x3024, processed at 2736x2048 in 15 tiles)" }, error: null },
        { assetId: "b", plan: null, error: "needs a 16128x12096 output" },
      ],
    };
    const summary = summarizeRestorePlans(response);
    expect(summary.blocked).toBe(true);
    expect(summary.lines[0]).toMatch(/^budgeted: 683x512 input → 2732x2048/);
    expect(summary.lines[0]).not.toMatch(/of the original/);
    expect(summary.lines[1]).toBe("Cannot restore: needs a 16128x12096 output");
    expect(summary.firstError).toBe("needs a 16128x12096 output");
    expect(summarizeRestorePlans({ items: [response.items[0]] }).blocked).toBe(false);
    expect(restorePlanItemText(null)).toBe("");
  });

  it("reopens applied settings from an Iris record only", () => {
    expect(irisControlsFromRecord({ engine: "iris", scale: 1.5, inputSizing: "original", colorFix: false })).toEqual({
      scale: 1.5,
      inputSizing: "original",
      colorFix: false,
    });
    expect(irisControlsFromRecord({ engine: "seedvr2", factor: 2 })).toBeNull();
    expect(isIrisEngine("IRIS")).toBe(true);
  });
});

describe("Iris restoration job bodies", () => {
  it("sends scale / input sizing / colour fix — never a factor or softness", () => {
    expect(upscaleEngineIsRestoration("iris")).toBe(true);
    expect(upscaleEngineIsRestoration("seedvr2")).toBe(false);
    const body = buildUpscaleJobBody({
      project,
      requestedGpu: "auto",
      sourceAssetId: "s1",
      factor: 4,
      engine: "iris",
      displayName: "Photo",
      softness: 0.5,
      restore: { scale: 1, inputSizing: "original", colorFix: false },
    });
    expect(body).toEqual({
      type: "image_upscale",
      projectId: "p1",
      projectName: "Project",
      requestedGpu: "auto",
      payload: {
        projectId: "p1",
        sourceAssetId: "s1",
        engine: "iris",
        displayName: "Photo",
        scale: 1,
        inputSizing: "original",
        colorFix: false,
      },
    });
  });

  it("fans the Iris controls out per asset for a Library batch", () => {
    const job = buildBatchJob({
      op: "upscale",
      asset: { id: "a1", displayName: "One" },
      params: { engine: "iris", restore: { scale: 2, inputSizing: "budgeted", colorFix: true } },
      project,
      requestedGpu: "auto",
    });
    expect(job.endpoint).toBe("/api/v1/jobs");
    expect(job.body.payload).toEqual({
      projectId: "p1",
      sourceAssetId: "a1",
      engine: "iris",
      displayName: "One",
      scale: 2,
      inputSizing: "budgeted",
      colorFix: true,
    });
  });

  it("leaves the factor engines' bodies unchanged", () => {
    const body = buildUpscaleJobBody({
      project,
      requestedGpu: "auto",
      sourceAssetId: "s1",
      factor: 2,
      engine: "real-esrgan",
      displayName: "Photo",
      restore: { scale: 3 },
    });
    expect(body.payload).toEqual({ projectId: "p1", sourceAssetId: "s1", factor: 2, engine: "real-esrgan", displayName: "Photo" });
  });
});

describe("trained Iris models (sc-25687 / sc-25688)", () => {
  const models = [
    { id: "iris_3b_restore", installState: "installed" },
    { id: "mine_restore", installState: "installed", trainedCheckpoint: { task: "restoration" } },
    { id: "pending_restore", installState: "missing", trainedCheckpoint: { task: "restoration" } },
    { id: "mine_depth", installState: "installed", trainedCheckpoint: { task: "depth" } },
  ];

  it("lists only installed trained models of the task", () => {
    expect(trainedTaskModels(models, "restoration").map((model) => model.id)).toEqual(["mine_restore"]);
    expect(trainedTaskModels(models, "depth").map((model) => model.id)).toEqual(["mine_depth"]);
  });

  it("carries irisModel beside the Iris controls — omitted for the release export", () => {
    expect(irisRestoreControls({ scale: 2, irisModel: "mine_restore" }).irisModel).toBe("mine_restore");
    expect(irisRestoreControls({ irisModel: "iris_3b_restore" })).not.toHaveProperty("irisModel");
    expect(irisRestoreControls({ irisModel: "" })).not.toHaveProperty("irisModel");
    expect(irisControlsFromRecord({ engine: "iris", scale: 4, irisModel: "mine_restore" }).irisModel).toBe("mine_restore");
    const job = buildBatchJob({
      op: "upscale",
      asset: { id: "a1" },
      params: { engine: "iris", restore: irisRestoreControls({ irisModel: "mine_restore" }) },
      project,
      requestedGpu: "auto",
    });
    expect(job.body.payload.irisModel).toBe("mine_restore");
  });
});

// S2: a Studio prompt batch plans every distinct render size — each prompt's own [WxH] directive, or
// the Studio size for a prompt without one — instead of only the Studio default.
describe("Iris restore plan body for Studio renders (S2)", () => {
  it("plans the Studio size for a single render or a directive-free batch", () => {
    expect(restorePlanBodyForRenders({ width: 1024, height: 768 })).toEqual({ width: 1024, height: 768 });
    expect(restorePlanBodyForRenders({ width: 1024, height: 768, resolutions: [null, null] })).toEqual({
      width: 1024,
      height: 768,
    });
  });

  it("plans every distinct per-prompt size, deduplicated, alongside the Studio size", () => {
    expect(
      restorePlanBodyForRenders({
        width: 1024,
        height: 768,
        resolutions: [{ width: 768, height: 1344 }, null, { width: 768, height: 1344 }],
      }),
    ).toEqual({ sizes: [{ width: 768, height: 1344 }, { width: 1024, height: 768 }] });
  });

  it("plans a batch whose every prompt overrides the size at that size only", () => {
    expect(
      restorePlanBodyForRenders({ width: 1024, height: 768, resolutions: [{ width: 768, height: 1344 }] }),
    ).toEqual({ width: 768, height: 1344 });
  });

  it("has nothing to plan without dimensions", () => {
    expect(restorePlanBodyForRenders({ width: 0, height: 768 })).toBeNull();
  });
});
