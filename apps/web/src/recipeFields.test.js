import { describe, expect, it } from "vitest";

import {
  finiteRecipeNumber,
  recipeLoraSelection,
  recipeRequestedResolution,
  recipeResolution,
} from "./recipeFields.js";

describe("finiteRecipeNumber", () => {
  it("keeps finite numbers (including 0) and rejects the rest", () => {
    expect(finiteRecipeNumber(0)).toBe(0);
    expect(finiteRecipeNumber("768")).toBe(768);
    expect(finiteRecipeNumber(undefined)).toBeNull();
    expect(finiteRecipeNumber("wide")).toBeNull();
    expect(finiteRecipeNumber(Infinity)).toBeNull();
  });

  // Documenting a sharp edge rather than asserting what you'd assume: `Number(null)` is 0, so a
  // null coerces to 0 instead of rejecting. Every caller here happens to be guarded — the
  // resolution readers test the result for truthiness, and 0 is falsy — but a new caller that
  // trusts a 0 would be reading "absent" as "zero".
  it("coerces null/empty to 0, NOT null", () => {
    expect(finiteRecipeNumber(null)).toBe(0);
    expect(finiteRecipeNumber("")).toBe(0);
  });
});

describe("recipeResolution", () => {
  it("prefers the resolved dims", () => {
    const recipe = {
      normalizedSettings: { width: 832, height: 480 },
      rawAdapterSettings: { resolution: "848x480" },
    };
    expect(recipeResolution(recipe)).toBe("832x480");
  });

  it("falls back to the requested string when dims are absent", () => {
    expect(recipeResolution({ rawAdapterSettings: { resolution: "1280x720" } })).toBe("1280x720");
  });

  it("returns null when the recipe carries neither", () => {
    expect(recipeResolution({})).toBeNull();
    expect(recipeResolution(undefined)).toBeNull();
    expect(recipeResolution({ rawAdapterSettings: { resolution: "wide" } })).toBeNull();
  });
});

describe("recipeRequestedResolution", () => {
  // The reason this reader exists: Video Studio's <select> only holds `limits.resolutions`
  // options. A clip requested at 848x480 that the stride floor resolved to 832x480 must replay
  // the option the user picked, or the control snaps away to the model default.
  it("prefers the requested string over the resolved dims", () => {
    const recipe = {
      normalizedSettings: { width: 832, height: 480 },
      rawAdapterSettings: { resolution: "848x480" },
    };
    expect(recipeRequestedResolution(recipe)).toBe("848x480");
  });

  it("falls back to the resolved dims when advanced was never recorded", () => {
    expect(recipeRequestedResolution({ normalizedSettings: { width: 768, height: 512 } })).toBe(
      "768x512",
    );
  });

  it("returns null when the recipe carries neither", () => {
    expect(recipeRequestedResolution({})).toBeNull();
  });
});

describe("recipeLoraSelection", () => {
  it("reads both the bare-id and the {id, weight} shapes", () => {
    const { loraIds, loraWeights } = recipeLoraSelection({
      loras: ["lora_bare", { id: "lora_a", weight: 0.8 }, { loraId: "lora_b", weight: 0.5 }],
    });
    expect(loraIds).toEqual(["lora_bare", "lora_a", "lora_b"]);
    expect(loraWeights).toEqual({ lora_a: 0.8, lora_b: 0.5 });
  });

  it("selects a weightless LoRA without inventing a weight", () => {
    const { loraIds, loraWeights } = recipeLoraSelection({ loras: [{ id: "lora_a" }] });
    expect(loraIds).toEqual(["lora_a"]);
    expect(loraWeights).toEqual({});
  });

  it("keeps an explicit weight of 0 rather than dropping it", () => {
    expect(recipeLoraSelection({ loras: [{ id: "lora_a", weight: 0 }] }).loraWeights).toEqual({
      lora_a: 0,
    });
  });

  it("tolerates a recipe with no LoRAs", () => {
    expect(recipeLoraSelection({})).toEqual({ loraIds: [], loraWeights: {} });
    expect(recipeLoraSelection({ loras: null })).toEqual({ loraIds: [], loraWeights: {} });
  });
});
