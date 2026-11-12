import { describe, expect, it } from "vitest";
import { fitsResolutionOptions, resolutionFitsMemory } from "./resolutionMemory.js";
import candidate from "./testUtils/catalogBaseline/candidate-catalog.json";

// The resolution gate reads the API-served thresholds (`memoryGuidance.<lane>.resolutionMinHostGb`,
// SW-7032). These tests drive the shipped native catalog fixture, so the expectations follow the
// API's prediction rather than a number re-derived here.
const shipped = (id) => structuredClone(candidate.models.find((model) => model.id === id));
const krea = () => shipped("krea_2_raw");
const KREA_RES = krea().limits.resolutions;
const withInstalled = (model, tiers) => ({
  ...model,
  hasVariantMatrix: true,
  variants: tiers.map((variant) => ({ variant, installState: "installed" })),
});

describe("resolutionFitsMemory", () => {
  it("always offers buckets at or below the historical 1536² ceiling", () => {
    for (const res of ["1024x1024", "1536x1536", "2048x1152"]) {
      expect(resolutionFitsMemory(krea(), res, 1, { backend: "mlx" })).toBe(true);
      expect(resolutionFitsMemory(krea(), res, 1, { backend: "candle" })).toBe(true);
    }
  });

  it("never withholds on an unknown host reading", () => {
    for (const host of [null, undefined, NaN]) {
      expect(resolutionFitsMemory(krea(), "2048x2048", host, { backend: "mlx" })).toBe(true);
    }
  });

  it("hides 2048² on a host below the served threshold, per lane", () => {
    expect(resolutionFitsMemory(krea(), "2048x2048", 48, { backend: "mlx" })).toBe(false);
    expect(resolutionFitsMemory(krea(), "2048x2048", 128, { backend: "mlx" })).toBe(true);
    expect(resolutionFitsMemory(krea(), "2048x2048", 48, { backend: "candle" })).toBe(false);
    expect(resolutionFitsMemory(krea(), "2048x2048", 80, { backend: "candle" })).toBe(true);
    expect(resolutionFitsMemory(krea(), "2048x2048", 64, { backend: "mlx" })).toBe(false);
  });

  it("admits exactly at the served threshold", () => {
    const model = krea();
    const threshold = model.memoryGuidance.mlx.resolutionMinHostGb["2048x2048"];
    expect(resolutionFitsMemory(model, "2048x2048", threshold, { backend: "mlx" })).toBe(true);
    expect(resolutionFitsMemory(model, "2048x2048", threshold - 0.001, { backend: "mlx" })).toBe(false);
  });

  it("follows a changed served threshold with no web change", () => {
    const model = krea();
    model.memoryGuidance.mlx.resolutionMinHostGb["2048x2048"] = 32;
    expect(resolutionFitsMemory(model, "2048x2048", 48, { backend: "mlx" })).toBe(true);
  });

  it("reads an installed tier's own thresholds and ignores a stale or uninstalled tier", () => {
    const model = withInstalled(krea(), ["q4"]);
    model.memoryGuidance.candle.tiers.q4.resolutionMinHostGb = { "2048x2048": 40 };
    model.memoryGuidance.candle.tiers.q8.resolutionMinHostGb = { "2048x2048": 20 };
    expect(resolutionFitsMemory(model, "2048x2048", 41, { backend: "candle", tier: "q4" })).toBe(true);
    expect(resolutionFitsMemory(model, "2048x2048", 39, { backend: "candle", tier: "q4" })).toBe(false);
    // q8 is not installed: the lane-level threshold applies, never q8's evidence.
    const lane = model.memoryGuidance.candle.resolutionMinHostGb["2048x2048"];
    expect(resolutionFitsMemory(model, "2048x2048", lane - 1, { backend: "candle", tier: "q8" })).toBe(false);
    expect(resolutionFitsMemory(model, "2048x2048", lane - 1, { backend: "candle", tier: "stale" })).toBe(false);
  });

  it("is a no-op before the API serves guidance or for a model with no floor on the lane", () => {
    const unserved = krea();
    delete unserved.memoryGuidance;
    expect(resolutionFitsMemory(unserved, "2048x2048", 8, { backend: "mlx" })).toBe(true);
    const sensenova = shipped("sensenova_u1_8b");
    expect(resolutionFitsMemory(sensenova, "2048x2048", 16, { backend: "mlx" })).toBe(true);
  });
});

describe("fitsResolutionOptions", () => {
  it("trims only the over-budget high-res buckets, preserving order", () => {
    const filtered = fitsResolutionOptions(krea(), KREA_RES, 48, { backend: "mlx" });
    expect(filtered).not.toContain("2048x2048");
    expect(filtered).toContain("1536x1536");
    expect(filtered).toEqual(KREA_RES.filter((res) => filtered.includes(res)));
    expect(fitsResolutionOptions(krea(), KREA_RES, 256, { backend: "mlx" })).toEqual(KREA_RES);
    expect(fitsResolutionOptions(krea(), KREA_RES, null, { backend: "mlx" })).toEqual(KREA_RES);
    expect(fitsResolutionOptions(krea(), undefined, 48, { backend: "mlx" })).toEqual([]);
  });
});
