import { describe, expect, it } from "vitest";
import {
  MAX_IMAGE_DIMENSION,
  MIN_IMAGE_DIMENSION,
  resolveEffectiveDimensions,
} from "./resolutionOverride.js";

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
