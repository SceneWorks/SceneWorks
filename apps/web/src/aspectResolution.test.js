import { describe, expect, it } from "vitest";
import {
  ASPECT_RATIOS,
  aspectRatiosFor,
  fitResolutionToEnvelope,
  nearestAspectRatio,
  resolutionEnvelope,
  resolutionFitsEnvelope,
  resolutionFor,
  stepMegapixels,
} from "./aspectResolution.js";

// Ladders copied from config/manifests/builtin.models.jsonc, so the derived bands are the shipped ones.
const KREA_2 = ["1024x1024", "896x896", "768x768", "768x1024", "1024x768", "1280x720", "720x1280",
  "1216x832", "832x1216", "1152x896", "896x1152", "1536x1536", "2048x1152", "1152x2048", "2048x1408",
  "1408x2048", "2048x2048"];
const Z_IMAGE = ["768x768", "1024x1024", "1280x720", "720x1280"];
const H3_IMAGE = ["2048x2048", "4096x4096", "4736x3552", "3552x4736", "5024x3360", "3360x5024",
  "5472x3072", "3072x5472", "6272x2688", "6144x6144", "7104x5344", "5344x7104", "7552x5056",
  "5056x7552", "8224x4608", "4608x8224", "9408x4032"];
const H3_MODEL = { limits: { resolutions: H3_IMAGE, minDimension: 16, maxDimension: 9408 } };
const QWEN_2_1 = { limits: { resolutions: ["2048x2048", "2752x1536", "1536x2752", "2368x1760", "1760x2368", "2496x1664", "1664x2496"],
  minDimension: 32, maxDimension: 2752, requiresDimensionsMultipleOf: 32 } };
const MAGE = { limits: { resolutions: ["1024x1024", "2048x2048", "2048x512", "512x2048", "1536x768", "768x1536"],
  requiresDimensionsMultipleOf: 16 } };

const envelopeFor = (resolutions, limits = {}) =>
  resolutionEnvelope({ limits: { resolutions, ...limits } }, resolutions);

describe("resolutionEnvelope", () => {
  it("bands each model between its smallest and largest preset area", () => {
    expect(envelopeFor(KREA_2).maxMegapixels).toBe(4);
    expect(envelopeFor(KREA_2).minMegapixels).toBe(0.5625);
    expect(envelopeFor(Z_IMAGE).maxMegapixels).toBe(1);
    const h3 = resolutionEnvelope(H3_MODEL, H3_IMAGE);
    expect(h3.minMegapixels).toBe(4);
    expect(h3.maxMegapixels).toBeCloseTo(36.41, 2);
    expect(h3.maxSide).toBe(9408);
  });

  it("treats a single published size as fixed", () => {
    const supra = envelopeFor(["256x256"], { minDimension: 256, maxDimension: 256 });
    expect(supra.fixed).toBe(true);
    expect(resolutionFor("16:9", 4, supra)).toBe("256x256");
    expect(resolutionFitsEnvelope("256x256", supra)).toBe(true);
    expect(resolutionFitsEnvelope("512x512", supra)).toBe(false);
  });

  it("uses the declared grid when coarser than 16", () => {
    expect(resolutionEnvelope(QWEN_2_1, QWEN_2_1.limits.resolutions).stride).toBe(32);
  });
});

describe("resolutionFor", () => {
  it("derives the ComfyUI-style sizes on Krea 2 without exceeding its 2048² area", () => {
    const krea = envelopeFor(KREA_2);
    expect(resolutionFor("1:1", 4, krea)).toBe("2048x2048");
    expect(resolutionFor("3:4", 4, krea)).toBe("1776x2352");
    expect(resolutionFor("16:9", 4, krea)).toBe("2720x1536");
    expect(resolutionFor("1:1", 1, krea)).toBe("1024x1024");
    for (const ratio of ASPECT_RATIOS) {
      const [w, h] = resolutionFor(ratio, 99, krea).split("x").map(Number);
      expect(w * h).toBeLessThanOrEqual(2048 * 2048);
      expect(w % 16 + h % 16).toBe(0);
    }
  });

  it("clamps to the band: Z-Image tops out at 1 MP, H3 starts at 4 MP", () => {
    expect(resolutionFor("16:9", 4, envelopeFor(Z_IMAGE))).toBe("1360x768");
    expect(resolutionFor("1:1", 1, resolutionEnvelope(H3_MODEL, H3_IMAGE))).toBe("2048x2048");
  });

  it("reaches H3's published 21:9 presets exactly and never passes its 9408 edge", () => {
    const h3 = resolutionEnvelope(H3_MODEL, H3_IMAGE);
    expect(resolutionFor("21:9", 6272 * 2688 / (1024 * 1024), h3)).toBe("6272x2688");
    expect(resolutionFor("21:9", 99, h3)).toBe("9408x4032");
  });

  it("keeps Qwen-Image 2.1 square, on its 32 px grid and inside its 2752 edge", () => {
    const qwen = resolutionEnvelope(QWEN_2_1, QWEN_2_1.limits.resolutions);
    expect(resolutionFor("1:1", 99, qwen)).toBe("2048x2048");
    expect(resolutionFor("16:9", 99, qwen)).toBe("2752x1536");
  });
});

describe("model preset shapes", () => {
  it("adds Mage-Flow's panorama shapes and nothing for standard ladders", () => {
    expect(aspectRatiosFor(KREA_2)).toHaveLength(ASPECT_RATIOS.length);
    expect(aspectRatiosFor(H3_IMAGE)).toHaveLength(ASPECT_RATIOS.length);
    const mage = resolutionEnvelope(MAGE, MAGE.limits.resolutions);
    expect(mage.ratios.slice(ASPECT_RATIOS.length).map((ratio) => ratio.id)).toEqual(["4:1", "2:1", "1:2", "1:4"]);
    expect(resolutionFor("4:1", 4, mage)).toBe("2048x512");
  });

  it("names a grid-aligned ladder size by its nearest standard ratio", () => {
    expect(nearestAspectRatio(1216, 832).id).toBe("3:2");
    expect(nearestAspectRatio(1152, 896).id).toBe("4:3");
    expect(nearestAspectRatio(720, 1280).id).toBe("9:16");
  });
});

describe("switching models", () => {
  it("keeps a size the new model can render and refits one it cannot, keeping the ratio", () => {
    const krea = envelopeFor(KREA_2);
    const zImage = envelopeFor(Z_IMAGE);
    expect(resolutionFitsEnvelope("1776x2352", krea)).toBe(true);
    expect(resolutionFitsEnvelope("1776x2352", zImage)).toBe(false);
    expect(fitResolutionToEnvelope("1776x2352", zImage)).toBe("880x1184");
    // A preset of another model that this one can render is not rewritten.
    expect(resolutionFitsEnvelope("1216x832", krea)).toBe(true);
  });

  it("enforces a declared grid but not an undeclared one", () => {
    const mage = resolutionEnvelope(MAGE, MAGE.limits.resolutions);
    expect(resolutionFitsEnvelope("1000x1000", mage)).toBe(false);
    expect(resolutionFitsEnvelope("1000x1000", envelopeFor(KREA_2))).toBe(true);
  });
});

describe("stepMegapixels", () => {
  it("steps on a grid sized to the model's range and stops at the band edges", () => {
    const krea = envelopeFor(KREA_2);
    expect(stepMegapixels(3.98, 1, krea)).toBe(4);
    expect(stepMegapixels(1, 1, krea)).toBe(1.25);
    expect(stepMegapixels(0.96, -1, krea)).toBe(0.75);
    expect(stepMegapixels(0.6, -1, krea)).toBe(0.5625);
    const h3 = resolutionEnvelope(H3_MODEL, H3_IMAGE);
    expect(stepMegapixels(4, 1, h3)).toBe(5);
  });
});
