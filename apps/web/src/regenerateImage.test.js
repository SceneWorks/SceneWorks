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

const mossModel = { id: "moss_image_base", conditionalPurposes: { moss_image_edit: { installState: "installed" } } };
const mossRefs = Array.from({ length: 4 }, (_, i) => ({ id: `ref-${i}`, type: "image", projectId: "project-1" }));
const tensor = (id, dtype, shape) => ({ id, type: "tensor", projectId: "project-1", tensor: { dtype, shape, byteLength: shape.reduce((a, b) => a * b, dtype === "float32" ? 4 : 1), sha256: "a".repeat(64) } });
const positive = tensor("positive", "float32", [2, 2, 4096]), mask = tensor("mask", "uint8", [2, 2]), noise = tensor("noise", "float32", [4, 128, 32, 32]);
const ref = (item) => ({ assetId: item.id, sha256: item.tensor.sha256, metadata: item.tensor });
const mossAssets = [...mossRefs, positive, mask, noise];
function mossAsset(output = "images") {
  return { id: "moss-output", projectId: "project-1", type: "image", lineage: { sourceAssetId: "unrelated-output" }, recipe: {
    model: mossModel.id, mode: "edit_image", prompt: "", seed: 999,
    normalizedSettings: { width: 512, height: 512, count: 2 },
    rawAdapterSettings: { referenceAssetIds: ["ref-0", "ref-1", "ref-0", "ref-3"], steps: 30, guidanceScale: 4, schedulerShift: 5,
      mossImage: { acceleration: "off", max_sequence_length: 2, training_shift: 2.5 }, mossImageEdit: { reference_max_edge: 512, use_kv_cache: false },
      mossImageAdvanced: { positive: { values: ref(positive), mask: ref(mask) }, negativePrompts: ["blur", "text"], noise: ref(noise), output },
      mossImageSampleIndex: 3, mossImageEditExecutionReport: { base: { samples: [{ seed: 173 }] } } } } };
}

describe("background MOSS edit regeneration", () => {
  it.each(["images", "latents"])("preserves the complete B2/count2 %s recipe without generic mask fields or output receipts", async (output) => {
    apiFetch.mockReset().mockResolvedValueOnce({ id: "new-moss" });
    const source = mossAsset(output), before = JSON.stringify(source), request = imageRecipeRequest(source);
    const opts = options({ asset: source, models: [mossModel], assets: mossAssets });
    expect(await regenerateImage(opts)).toEqual({ id: "new-moss" });
    const body = JSON.parse(apiFetch.mock.calls[0][2].body);
    expect(body).toMatchObject({ count: 2, seed: null, sourceAssetId: null, referenceAssetIds: ["ref-0", "ref-1", "ref-0", "ref-3"], advanced: { mossImageAdvanced: source.recipe.rawAdapterSettings.mossImageAdvanced, mossImage: { training_shift: 2.5 } } });
    expect(request.advanced).not.toHaveProperty("mossImageSampleIndex");
    expect(request.advanced).not.toHaveProperty("mossImageEditExecutionReport");
    expect(JSON.stringify(source)).toBe(before);
  });

  it("replays every prompt row from a later-row image with per-prompt count", () => {
    const source = mossAsset();source.recipe.prompt = "second row";
    source.recipe.rawAdapterSettings.promptBatch = ["first row", "second row"];
    delete source.recipe.rawAdapterSettings.mossImageAdvanced.positive;
    expect(imageRecipeRequest(source)).toMatchObject({ prompt: "first row", prompts: ["first row", "second row"], count: 2, seed: null });
  });

  it.each([true, false])("keeps supplied-noise seed null for the actual original-job duplicate (noise: %s)", async (withNoise) => {
    const source = mossAsset();source.lineage.jobId = "original";
    if (!withNoise) delete source.recipe.rawAdapterSettings.mossImageAdvanced.noise;
    const payload = imageRecipeRequest(source);
    apiFetch.mockReset().mockResolvedValueOnce({ id: "original", type: "image_generate", projectId: "project-1", payload }).mockResolvedValueOnce({ id: "new-moss" });
    await regenerateImage(options({ asset: source, models: [mossModel], assets: mossAssets }));
    const changes = JSON.parse(apiFetch.mock.calls[1][2].body).payloadChanges;
    expect(changes.seeds).toEqual([]);
    if (withNoise) expect(changes.seed).toBeNull();else expect(changes.seed).toEqual(expect.any(Number));
  });

  it.each(["missing adapter", "missing reference", "missing noise"])("rejects %s before either duplicate or fallback enqueue", async (issue) => {
    for (const originalAvailable of [true, false]) {
      const source = mossAsset();
      if (originalAvailable) source.lineage.jobId = "original";
      const model = issue === "missing adapter" ? { ...mossModel, conditionalPurposes: {} } : mossModel;
      const assets = mossAssets.filter((item) => item.id !== (issue === "missing reference" ? "ref-0" : issue === "missing noise" ? "noise" : "none"));
      apiFetch.mockReset();
      if (originalAvailable) apiFetch.mockResolvedValueOnce({ id: "original", type: "image_generate", projectId: "project-1", payload: imageRecipeRequest(source) });
      const opts = options({ asset: source, models: [model], assets });
      await expect(regenerateImage(opts)).rejects.toThrow(issue === "missing adapter" ? "exact selected MOSS component" : issue === "missing reference" ? "reference is missing" : "noise tensor is missing");
      expect(apiFetch.mock.calls.filter((call) => call[2]?.method === "POST")).toHaveLength(0);
      expect(opts.onCreated).not.toHaveBeenCalled();
    }
  });
});
