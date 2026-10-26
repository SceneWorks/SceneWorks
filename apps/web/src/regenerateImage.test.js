import { afterEach, describe, expect, it, vi } from "vitest";
import { apiFetch } from "./api.js";
import { imageRecipeRequest, regenerateImage } from "./regenerateImage.js";

vi.mock("./api.js", () => ({ apiFetch: vi.fn() }));
afterEach(() => vi.restoreAllMocks());

const asset = {
  id: "image-1", projectId: "project-1", type: "image",
  lineage: { jobId: "original", sourceAssetId: "source" },
  recipe: { mode: "edit_image", model: "model", prompt: "Subject: lighthouse\nStyle: cinematic", seed: 123,
    loras: [{ id: "style", weight: 0.7 }],
    normalizedSettings: { width: 1024, height: 768, count: 2 },
    rawAdapterSettings: { steps: 28, guidanceScale: 3.5, referenceAssetIds: ["a", "b"],
      upscale: { enabled: true, factor: 2 }, hiresFix: { enabled: true, upscaleBy: 2 },
      sourceAssetId: "source", maskAssetId: null, fitMode: "crop",
      promptEnhancement: { receipt: true }, mossImageExecutionReport: { receipt: true } } },
};
function options(overrides = {}) {
  return { asset, projectId: "project-1", projectName: "Noir", token: "token", requestedGpu: "auto",
    beforeCreate: vi.fn(async () => true),
    isCurrentProject: () => true, onCreated: vi.fn(), ...overrides };
}

describe("background image regeneration", () => {
  it("duplicates the original job with fresh randomness instead of retaining its seed array", async () => {
    const random = vi.spyOn(crypto, "getRandomValues").mockImplementation((values) => { values[0] = 987; return values; });
    apiFetch.mockResolvedValueOnce({ id: "original", type: "image_edit", projectId: "project-1" })
      .mockResolvedValueOnce({ id: "new-job" });
    const opts = options();
    expect(await regenerateImage(opts)).toEqual({ id: "new-job" });
    expect(random).toHaveBeenCalledOnce();
    expect(JSON.parse(apiFetch.mock.calls[1][2].body)).toEqual({ requestedGpu: "auto", payloadChanges: { seed: 987, seeds: [] } });
    expect(apiFetch.mock.calls[1][0]).toBe("/api/v1/jobs/original/duplicate");
    expect(opts.beforeCreate).toHaveBeenCalledOnce();
    expect(opts.onCreated).toHaveBeenCalledWith({ id: "new-job" });
  });

  it.each([false, true])("replays recipes when the original job is missing (has job id: %s)", async (hasJobId) => {
    apiFetch.mockReset();
    if (hasJobId) apiFetch.mockRejectedValueOnce(Object.assign(new Error("gone"), { status: 404 }));
    apiFetch.mockResolvedValueOnce({ id: "fallback" });
    const opts = options({ asset: { ...asset, lineage: hasJobId ? asset.lineage : {} } });
    await regenerateImage(opts);
    const [path, , request] = apiFetch.mock.calls.at(-1);
    expect(path).toBe("/api/v1/image/jobs");
    const body = JSON.parse(request.body);
    expect(body).toMatchObject({ seed: null, model: "model", count: 2, width: 1024, height: 768 });
    expect(body).not.toHaveProperty("seeds");
    expect(opts.onCreated).toHaveBeenCalledWith({ id: "fallback" });
  });

  it("preserves recorded inputs and already-composed prompts without output receipts", () => {
    const request = imageRecipeRequest(asset);
    expect(request).toMatchObject({ prompt: asset.recipe.prompt, loras: asset.recipe.loras, sourceAssetId: "source",
      referenceAssetIds: ["a", "b"], upscale: { enabled: true, factor: 2 }, hiresFix: { enabled: true, upscaleBy: 2 },
      advanced: { steps: 28, guidanceScale: 3.5 } });
    expect(request.advanced).not.toHaveProperty("mossImageExecutionReport");
    expect(request.advanced).not.toHaveProperty("promptEnhancement");
    expect(asset.recipe.rawAdapterSettings).toHaveProperty("mossImageExecutionReport");
  });

  it("strips every receipt while preserving nested controls and the original recipes", () => {
    const original = { ...asset, generationSet: { recipe: { ...asset.recipe,
      rawAdapterSettings: { steps: 14, mossImage: { acceleration: "off" }, promptBatch: ["cat", "dog"],
        checkpointWeightFacts: { receipt: true }, memoryStrategy: "resident",
        memoryEvidenceRevision: "revision", realModelInference: true } } } };
    const before = JSON.stringify(original);
    const request = imageRecipeRequest(original);
    for (const key of ["promptBatch", "upscale", "hiresFix", "mossImageExecutionReport", "checkpointWeightFacts", "memoryStrategy", "memoryEvidenceRevision", "realModelInference", "promptEnhancement"]) {
      expect(request.advanced).not.toHaveProperty(key);
    }
    expect(request).toMatchObject({ prompts: ["cat", "dog"], upscale: { enabled: true, factor: 2 },
      hiresFix: { enabled: true, upscaleBy: 2 }, advanced: { steps: 28, mossImage: { acceleration: "off" } } });
    expect(JSON.stringify(original)).toBe(before);
  });

  it("preserves native prompt batches and their per-prompt count", () => {
    const request = imageRecipeRequest({ ...asset, generationSet: { recipe: { ...asset.recipe,
      rawAdapterSettings: { promptBatch: ["cat", "dog"], mossImage: { acceleration: "off" } } } } });
    expect(request).toMatchObject({ prompt: "cat", prompts: ["cat", "dog"], count: 2, seed: null });
    expect(request.advanced).not.toHaveProperty("promptBatch");
  });

  it("preserves per-asset processing controls on top of shared generation settings", () => {
    const request = imageRecipeRequest({ ...asset, generationSet: { recipe: { ...asset.recipe,
      rawAdapterSettings: { steps: 14 } } } });
    expect(request).toMatchObject({ upscale: { enabled: true, factor: 2 }, advanced: { steps: 28 } });
  });

  it("refuses incomplete historical edit inputs rather than queueing a different image", () => {
    expect(() => imageRecipeRequest({ ...asset, recipe: { ...asset.recipe, rawAdapterSettings: {} } }))
      .toThrow("lacks its complete source/mask settings");
  });

  it("preserves the actual typed-route error, including unavailable inline character LoRAs", async () => {
    apiFetch.mockReset().mockRejectedValueOnce(new Error("LoRA not found: character-adapter"));
    const opts = options({ asset: { ...asset, lineage: {} } });
    await expect(regenerateImage(opts)).rejects.toThrow("LoRA not found: character-adapter");
    expect(opts.onCreated).not.toHaveBeenCalled();
  });

  it("does not enqueue if preflight is canceled or the active project changes", async () => {
    apiFetch.mockReset().mockResolvedValue({ id: "original", projectId: "project-1", type: "image_generate" });
    const opts = options({ beforeCreate: vi.fn(async () => false) });
    expect(await regenerateImage(opts)).toBeNull();
    expect(await regenerateImage(options({ isCurrentProject: () => false }))).toBeNull();
    expect(apiFetch).toHaveBeenCalledTimes(2); // GETs only
  });

  it("propagates failures for the viewer to display without falling back to a studio", async () => {
    apiFetch.mockReset().mockRejectedValue(new Error("Host unavailable"));
    const opts = options();
    await expect(regenerateImage(opts)).rejects.toThrow("Host unavailable");
    expect(opts.onCreated).not.toHaveBeenCalled();
  });
});
