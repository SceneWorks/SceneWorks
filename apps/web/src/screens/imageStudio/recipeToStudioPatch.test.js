import { describe, expect, it } from "vitest";
import { recipeToStudioPatch } from "./recipeToStudioPatch.js";
const context = { model: "old", advancedDefaultsModel: "old", mossEditActive: false,
  recipeMode: (recipe) => recipe.mode ?? "text_to_image", ipAdapterScale: 0.7,
  controlnetScale: 0.8, trueCfgScale: 4, img2imgStrength: 0.5, textStyleGain: 1, controlScale: 1 };
const launch = (rawAdapterSettings = {}, extra = {}) => ({ id: "launch", view: "Image", replaySeed: 0,
  recipe: { model: "new", prompt: "composed prompt", rawAdapterSettings, normalizedSettings: {}, ...extra } });

describe("recipeToStudioPatch", () => {
  it("normalizes a replay without mutating its source or clearing absent reference selections", () => {
    const request = launch({ styleId: "style", stylePrompt: "original prompt", strength: 0, usePid: true });
    const before = structuredClone(request);
    const { values, flags } = recipeToStudioPatch(request, context);
    expect(request).toEqual(before);
    expect(values).toMatchObject({ prompt: "original prompt", seed: "0", styleId: "style", img2imgStrength: 0,
      ipAdapterScale: 0.7, usePid: true, textEncoderModel: { value: null, modelId: "new" } });
    expect(values).not.toHaveProperty("referenceAssetIds");
    expect(flags).toMatchObject({ skipReferenceTuningReset: true, skipAdvancedDefaultsReset: true,
      referenceTuningDeclaredArmed: false, promptEdited: true });
    expect(recipeToStudioPatch({ view: "Video" }, context)).toBeNull();
  });
  it("keeps explicit empty references and same-model reset ownership", () => {
    const { values, flags } = recipeToStudioPatch(launch({}, { model: "old", normalizedSettings: { referenceAssetIds: [] } }), context);
    expect(values.referenceAssetIds).toEqual([]);
    expect(flags.skipReferenceTuningReset).toBe(false);
    expect(flags.skipAdvancedDefaultsReset).toBeUndefined();
  });
  it("keeps engine and factor application order in the patch", () => {
    const { values } = recipeToStudioPatch(launch({ upscale: { enabled: true, factor: 4, engine: "real_esrgan", softness: 0.5 } }), context);
    expect(Object.keys(values).filter((key) => key.startsWith("upscale")))
      .toEqual(["upscaleEnabled", "upscaleFactor", "upscaleEngine", "upscaleSoftness"]);
  });
  it("restores raw appearance inputs and multiple look IDs rather than already composed text", () => {
    const appearance = { camera: null, film: "film-kodak-tri-x-400", looks: ["look-moody-dark", "look-light-leaks"], lighting: null };
    const { values } = recipeToStudioPatch(launch({ appearance, appearancePrompt: "Subject: a dancer", appearanceNegativePrompt: "authored", styleId: "style" }, { negativePrompt: "authored, recipe negatives" }), context);
    expect(values.appearance).toEqual(appearance);
    expect(values.prompt).toBe("Subject: a dancer");
    expect(values.negativePrompt).toBe("authored");
    expect(values.styleId).toBe("style");
    const old = recipeToStudioPatch(launch({ appearance }, {}), context).values;
    expect(old.appearance.looks).toEqual([]);
    expect(old.prompt).toBe("composed prompt");
  });

  it("restores an appearance batch from its authored carrier, preserving the worker's effective batch", () => {
    const { values } = recipeToStudioPatch(launch({ appearance: { looks: ["look-moody-dark"] }, appearancePrompt: "a fox",
      appearancePromptBatch: ["a fox", "a cat"], promptBatch: ["Description: a fox\nLook: dark", "Description: a cat\nLook: dark"],
      mossImageAdvanced: { output: "images" }, styleId: "style" }, { model: "moss_image_base", mode: "edit_image" }), context);
    expect(values.mossPromptBatch).toEqual(["a fox", "a cat"]);
    expect(values.prompt).toBe("a fox");
    expect(values.styleId).toBe("style");
    expect(values.appearance.looks).toEqual(["look-moody-dark"]);
  });

});
