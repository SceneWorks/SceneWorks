import { describe, expect, it } from "vitest";
import {
  MAX_IMAGE_DIMENSION,
  MIN_IMAGE_DIMENSION,
  dimensionConstraintMessage,
  evaluateModelDimensions,
  modelDimensionConstraints,
  resolveEffectiveDimensions,
} from "./resolutionOverride.js";

// A native-resolution model shaped like Mage-Flow's catalog projection: a ÷16 stride plus a
// resolution ladder whose extremes (512 and 2048) bound the free override.
const mageModel = {
  limits: {
    requiresDimensionsMultipleOf: 16,
    resolutions: ["512x512", "1024x1024", "2048x2048", "2048x512", "512x2048"],
  },
};
// A plain model with no stride — must be left on the global 256–4096 envelope, untouched.
const plainModel = { limits: { resolutions: ["1024x1024", "1280x720"] } };

describe("resolveEffectiveDimensions", () => {
  it("uses the Aspect dropdown when no override is set", () => {
    expect(
      resolveEffectiveDimensions({ resolution: "1024x1024", widthOverride: "", heightOverride: "" }),
    ).toEqual({ width: 1024, height: 1024, invalid: false });
  });

  it("overrides only the axis whose field is filled (per-axis)", () => {
    expect(
      resolveEffectiveDimensions({ resolution: "1024x768", widthOverride: "2048", heightOverride: "" }),
    ).toEqual({ width: 2048, height: 768, invalid: false });
    expect(
      resolveEffectiveDimensions({ resolution: "1024x768", widthOverride: "", heightOverride: "2048" }),
    ).toEqual({ width: 1024, height: 2048, invalid: false });
  });

  it("overrides both axes when both fields are filled (Krea 2 up to 4K)", () => {
    expect(
      resolveEffectiveDimensions({ resolution: "1024x1024", widthOverride: "4096", heightOverride: "4096" }),
    ).toEqual({ width: 4096, height: 4096, invalid: false });
  });

  it("accepts the exact backend bounds", () => {
    expect(
      resolveEffectiveDimensions({
        resolution: "1024x1024",
        widthOverride: String(MIN_IMAGE_DIMENSION),
        heightOverride: String(MAX_IMAGE_DIMENSION),
      }),
    ).toEqual({ width: MIN_IMAGE_DIMENSION, height: MAX_IMAGE_DIMENSION, invalid: false });
  });

  it("flags an override above the 4096 cap as invalid", () => {
    const result = resolveEffectiveDimensions({
      resolution: "1024x1024",
      widthOverride: "5000",
      heightOverride: "",
    });
    expect(result).toMatchObject({ width: 5000, height: 1024, invalid: true });
  });

  it("flags an override below the 256 floor as invalid", () => {
    const result = resolveEffectiveDimensions({
      resolution: "1024x1024",
      widthOverride: "",
      heightOverride: "64",
    });
    expect(result).toMatchObject({ width: 1024, height: 64, invalid: true });
  });

  it("treats null overrides like an empty field", () => {
    expect(
      resolveEffectiveDimensions({ resolution: "768x1280", widthOverride: null, heightOverride: null }),
    ).toEqual({ width: 768, height: 1280, invalid: false });
  });
});

describe("modelDimensionConstraints", () => {
  it("keeps a non-native model on the global 256–4096 envelope with no stride", () => {
    expect(modelDimensionConstraints(plainModel)).toEqual({
      min: MIN_IMAGE_DIMENSION,
      max: MAX_IMAGE_DIMENSION,
      step: 1,
    });
    // A missing model is the same untouched default (fallback catalog / loading state).
    expect(modelDimensionConstraints(undefined)).toEqual({
      min: MIN_IMAGE_DIMENSION,
      max: MAX_IMAGE_DIMENSION,
      step: 1,
    });
  });

  it("derives a native model's range from its ladder extremes and carries the stride", () => {
    expect(modelDimensionConstraints(mageModel)).toEqual({ min: 512, max: 2048, step: 16 });
  });
});

describe("evaluateModelDimensions (native-resolution enforcement)", () => {
  it("accepts a 4:1 native preset (2048x512) on a ÷16 model", () => {
    const result = evaluateModelDimensions({ model: mageModel, resolution: "2048x512" });
    expect(result).toMatchObject({ width: 2048, height: 512, invalid: false, offStride: false, outOfRange: false });
  });

  it("rejects an off-stride override as offStride (not out of range)", () => {
    const result = evaluateModelDimensions({
      model: mageModel,
      resolution: "1024x1024",
      widthOverride: "1000", // 1000 % 16 !== 0, but within 512–2048
      heightOverride: "",
    });
    expect(result).toMatchObject({ invalid: true, offStride: true, outOfRange: false });
  });

  it("rejects a below-native override (256) as out of range on a 512–2048 model", () => {
    const result = evaluateModelDimensions({
      model: mageModel,
      resolution: "1024x1024",
      widthOverride: "256", // ÷16 clean, but below the model's 512 native floor
      heightOverride: "",
    });
    expect(result).toMatchObject({ invalid: true, outOfRange: true, offStride: false });
  });

  it("rejects an above-native override (4096) as out of range even though the backend cap allows it", () => {
    const result = evaluateModelDimensions({
      model: mageModel,
      resolution: "1024x1024",
      widthOverride: "",
      heightOverride: "4096",
    });
    expect(result).toMatchObject({ invalid: true, outOfRange: true });
  });

  it("does NOT impose a stride on a plain model — a ÷16-violating size stays valid", () => {
    const result = evaluateModelDimensions({
      model: plainModel,
      resolution: "1024x1024",
      widthOverride: "1000",
      heightOverride: "",
    });
    expect(result).toMatchObject({ invalid: false, offStride: false });
  });
});

describe("optional reference axes do not loosen malformed geometry validation", () => {
  const flux3Model = { id: "flux3_klein", limits: { minDimension: 64, maxDimension: 4096,
    requiresDimensionsMultipleOf: 16, resolutions: ["1024x1024"] } };

  it.each([
    ["legacy model", plainModel, false, null, false],
    ["legacy model with unrelated intent", plainModel, true, { width: null, height: null }, false],
    ["plain FLUX3 T2I", flux3Model, false, null, true],
    ["plain FLUX3 T2I with unrelated intent", flux3Model, false, { width: null, height: null }, true],
    ["reference without intent", flux3Model, true, null, true],
    ["reference omitting only width", flux3Model, true, { width: null, height: 48 }, true],
  ])("rejects missing dropdown height for %s", (_label, model, referenceRequest, dimensionIntent, offStride) => {
    expect(evaluateModelDimensions({ model, referenceRequest, dimensionIntent,
      resolution: "1024", widthOverride: "", heightOverride: "" }))
      .toMatchObject({ height: undefined, invalid: true, outOfRange: true, offStride });
  });

  it("accepts a missing height only when that exact axis was deliberately omitted", () => {
    expect(evaluateModelDimensions({ model: flux3Model, referenceRequest: true,
      dimensionIntent: { width: 1024, height: null }, resolution: "1024",
      widthOverride: "", heightOverride: "" }))
      .toMatchObject({ width: 1024, height: undefined, invalid: false, outOfRange: false, offStride: false });
  });
});

describe("dimensionConstraintMessage", () => {
  it("returns null when the dimensions are valid", () => {
    const ok = evaluateModelDimensions({ model: mageModel, resolution: "1024x1024" });
    expect(dimensionConstraintMessage(ok)).toBeNull();
  });

  it("names the stride for an off-stride error", () => {
    const bad = evaluateModelDimensions({
      model: mageModel,
      resolution: "1024x1024",
      widthOverride: "1000",
    });
    expect(dimensionConstraintMessage(bad)).toBe(
      "Width and height must be a multiple of 16 — this model renders at native resolution.",
    );
  });

  it("names the native range for an out-of-range error", () => {
    const bad = evaluateModelDimensions({
      model: mageModel,
      resolution: "1024x1024",
      widthOverride: "256",
    });
    expect(dimensionConstraintMessage(bad)).toBe("Width and height must be between 512 and 2048.");
  });

  it("names both range and stride when both are broken", () => {
    const bad = evaluateModelDimensions({
      model: mageModel,
      resolution: "1024x1024",
      widthOverride: "5000", // above 2048 AND above the 4096 cap AND (5000 % 16 !== 0)
    });
    expect(dimensionConstraintMessage(bad)).toBe(
      "Width and height must be between 512 and 2048 and a multiple of 16.",
    );
  });
});
