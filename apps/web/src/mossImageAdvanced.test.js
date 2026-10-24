import { describe, it, expect } from "vitest";
import { mossAdvancedWire, mossTensorRef, reacquireMossTensors } from "./mossImageAdvanced.js";
import { mossAdvancedIssue } from "./mossImageAdvancedValidation.js";
import { mossSeededRecipeReplay, buildMossEditJobAdvanced } from "./mossImage.js";
import { recipeFromWorkflowShare } from "./workflowShare.js";
const artifact = (id, shape, dtype = "float32", sha = "a".repeat(64)) => ({ id, type: "tensor", tensor: { dtype, shape, byteLength: shape.reduce((a, b) => a * b, 1) * (dtype === "float32" ? 4 : 1), sha256: sha } });
describe("native edit artifact transport and replay", () => {
  it("preserves B2 × count2, exact noise identity, null seed and output choice through portable reacquisition", () => {
    const positive = artifact("original-positive", [2, 3, 4096]);
    const noise = artifact("original-noise", [4, 128, 1, 1], "float32", "b".repeat(64));
    const options = { positive: { values: mossTensorRef(positive) }, noise: mossTensorRef(noise), negativePrompts: ["blur", "text"], output: "latents" };
    expect(mossAdvancedIssue(options, { assets: [positive, noise], count: 2, width: 16, height: 16, context: 3 })).toBe("");
    const advanced = buildMossEditJobAdvanced({ mossImageAdvanced: options, mossImage: { max_sequence_length: 3, training_shift: 2.5 } });
    expect(advanced.mossImage.training_shift).toBe(2.5);
    const sharedOptions = structuredClone(options);
    delete sharedOptions.positive.values.assetId; delete sharedOptions.noise.assetId;
    const requirements = [["positive.values", positive], ["noise", noise]].map(([role, asset]) => ({ role, sha256: asset.tensor.sha256, metadata: asset.tensor }));
    const local = [{ ...positive, id: "local-positive" }, { ...noise, id: "local-noise" }];
    const picks = { "tensor:positive.values": "local-positive", "tensor:noise": "local-noise" };
    const share = { model: "moss_image_sft", mode: "edit_image", prompt: "", seed: null, width: 16, height: 16, count: 2, advanced: { ...advanced, mossImageAdvanced: sharedOptions, mossImageSampleIndex: 3 } };
    const recipe = recipeFromWorkflowShare(share, { model: { state: "resolved", catalogId: "moss_image_sft" }, tensors: requirements }, { tensorPicks: picks, assets: local });
    expect(recipe.normalizedSettings.count).toBe(2);
    expect(recipe.rawAdapterSettings.mossImageAdvanced).toEqual({ ...options, positive: { values: mossTensorRef(local[0]) }, noise: mossTensorRef(local[1]) });
    expect(recipe.rawAdapterSettings).not.toHaveProperty("mossImageSampleIndex");
    expect(mossSeededRecipeReplay({}, recipe, 123).replaySeed).toBeNull();
    expect(() => reacquireMossTensors(sharedOptions, requirements, picks, [local[0], { ...local[1], tensor: { ...noise.tensor, sha256: "c".repeat(64) } }])).toThrow(/exact noise/);
    expect(() => recipeFromWorkflowShare(share, { tensors: requirements }, { assets: local })).toThrow(/exact/);
  });
  it("rejects stale ownership, wrong shapes, conflicting carriers and unsupported knobs before submit", () => {
    const p = artifact("p", [2, 3, 4096]); const m = artifact("m", [2, 3], "uint8");
    const options = { positive: { values: mossTensorRef(p), mask: mossTensorRef(m) }, output: "images" };
    expect(mossAdvancedIssue(options, { assets: [], count: 1 })).toMatch(/missing or changed/);
    expect(mossAdvancedIssue(options, { prompt: "conflict" })).toMatch(/replace both/);
    expect(mossAdvancedIssue(options, { prompts: ["conflict"] })).toMatch(/replace both/);
    expect(mossAdvancedIssue(options, { context: 2 })).toMatch(/within the context/);
    expect(mossAdvancedIssue({ ...options, negativePrompts: ["only one"] })).toMatch(/match/);
    expect(() => mossAdvancedWire(0.75)).toThrow(/Unsupported/);
    expect(() => mossAdvancedWire({ inputLatents: {}, output: "images" })).toThrow(/Unsupported/);
    expect(() => mossAdvancedWire({ output: "png" })).toThrow(/images or latents/);
  });
  it("restores the full native text batch and chooses its base seed, preserving ordinary replay", () => {
    const recipe = { model: "moss_image_base", mode: "edit_image", normalizedSettings: { count: 2 }, rawAdapterSettings: { mossImageAdvanced: { output: "latents", negativePrompts: ["first", "second"] }, promptBatch: ["one", "two"], mossImageEditExecutionReport: { base: { samples: [{ seed: 70 }, { seed: 71 }] } } } };
    expect(mossSeededRecipeReplay({}, recipe, 73).replaySeed).toBe(70);
    expect(mossSeededRecipeReplay({}, { ...recipe, mode: "text_to_image", rawAdapterSettings: {} }, 73).replaySeed).toBe(73);
    expect(mossAdvancedIssue(recipe.rawAdapterSettings.mossImageAdvanced, { prompt: "one", prompts: ["one", "two"], count: 2 })).toBe("");
  });
  it.each(["images", "latents"])("replays non-first conditioning row %s output as the complete B2 count2 batch", (output) => {
    const noise = artifact("noise", [4, 128, 1, 1]);
    const raw = { promptBatch: ["first row", "second row"], referenceAssetIds: ["ref-0", "ref-1"], mossImageAdvanced: { output, negativePrompts: ["blur", "text"] }, mossImageSampleIndex: 2, mossImageEditExecutionReport: { base: { samples: [{ seed: 70 }, { seed: 71 }, { seed: 72 }, { seed: 73 }] } } };
    const recipe = { model: "moss_image_base", mode: "edit_image", prompt: "second row", seed: 72, normalizedSettings: { count: 2 }, rawAdapterSettings: raw };
    const asset = { type: output === "latents" ? "tensor" : "image", recipe, generationSet: { recipe: { seed: 70 } } };
    const original = structuredClone(recipe);
    const replay = mossSeededRecipeReplay(asset, recipe, 72);
    expect(replay).toEqual({ recipe: { ...recipe, prompt: "first row" }, replaySeed: 70 });
    expect(replay.recipe.rawAdapterSettings).toBe(raw);
    expect(recipe).toEqual(original);
    expect(mossAdvancedIssue(raw.mossImageAdvanced, { prompt: replay.recipe.prompt, prompts: raw.promptBatch, count: replay.recipe.normalizedSettings.count })).toBe("");
    expect(mossSeededRecipeReplay(asset, recipe, null).replaySeed).toBeNull();
    const callerNoise = { ...recipe, rawAdapterSettings: { ...raw, mossImageAdvanced: { ...raw.mossImageAdvanced, noise: mossTensorRef(noise) } } };
    expect(mossSeededRecipeReplay(asset, callerNoise, 72)).toEqual({ recipe: { ...callerNoise, prompt: "first row" }, replaySeed: null });
  });
  it("refuses inline upscale for latent output while accepting latent-only and decoded image upscale", () => {
    expect(mossAdvancedIssue({ output: "latents" }, { prompt: "edit", upscaleEnabled: true })).toMatch(/Disable upscale or select images/);
    expect(mossAdvancedIssue({ output: "latents" }, { prompt: "edit", upscaleEnabled: false })).toBe("");
    expect(mossAdvancedIssue({ output: "images" }, { prompt: "edit", upscaleEnabled: true })).toBe("");
  });

});
