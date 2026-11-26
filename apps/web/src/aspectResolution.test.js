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
const MAGE = { limits: { resolutions: ["512x512", "1024x1024", "2048x2048", "2048x512", "512x2048", "1536x768", "768x1536"],
  requiresDimensionsMultipleOf: 16 } };
const SDXL = ["1024x1024", "1152x896", "896x1152", "1216x832", "832x1216", "1344x768", "768x1344"];
// SenseNova's engine caps each side at 2048 and clamps a longer one, changing the shape (sc-12384).
const SENSENOVA = ["2048x2048", "2048x1152", "1152x2048", "1888x1248", "1248x1888", "1760x1312", "1312x1760"];

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

  it("reads the grid off the presets, which the worker pins to each engine's lattice", () => {
    expect(resolutionEnvelope(QWEN_2_1, QWEN_2_1.limits.resolutions).stride).toBe(32);
    expect(envelopeFor(KREA_2).stride).toBe(16);
    expect(envelopeFor(SENSENOVA).stride).toBe(32);
    // SDXL/Kolors are ÷32 on MLX: their presets' 64 grid covers it on every backend.
    expect(envelopeFor(SDXL).stride).toBe(64);
  });

  it("bounds sides by the presets unless the model declares an envelope", () => {
    expect(envelopeFor(KREA_2)).toMatchObject({ minSide: 720, maxSide: 2048 });
    expect(resolutionEnvelope(H3_MODEL, H3_IMAGE)).toMatchObject({ minSide: 32, maxSide: 9408 });
  });
});

describe("resolutionFor", () => {
  it("derives Krea 2 sizes inside its 2048² area and its engine's 2048 side", () => {
    const krea = envelopeFor(KREA_2);
    expect(resolutionFor("1:1", 4, krea)).toBe("2048x2048");
    expect(resolutionFor("3:4", 4, krea)).toBe("1536x2048");
    expect(resolutionFor("16:9", 4, krea)).toBe("2048x1152");
    expect(resolutionFor("21:9", 4, krea)).toBe("2048x880");
    expect(resolutionFor("1:1", 1, krea)).toBe("1024x1024");
    for (const ratio of ASPECT_RATIOS) {
      const [w, h] = resolutionFor(ratio, 99, krea).split("x").map(Number);
      expect(w * h).toBeLessThanOrEqual(2048 * 2048);
      expect(Math.max(w, h)).toBeLessThanOrEqual(2048);
      expect(w % 16 + h % 16).toBe(0);
    }
  });

  it("keeps SenseNova's 16:9 at its 2048 side cap rather than a clamped 4:3", () => {
    expect(resolutionFor("16:9", 99, envelopeFor(SENSENOVA))).toBe("2048x1152");
  });

  it("lands every SDXL size on the 32 px MLX lattice", () => {
    const sdxl = envelopeFor(SDXL);
    for (const ratio of sdxl.availableRatios) {
      for (const megapixels of [0, 0.98, 99]) {
        const [w, h] = resolutionFor(ratio, megapixels, sdxl).split("x").map(Number);
        expect(w % 32 + h % 32).toBe(0);
      }
    }
  });

  it("clamps to the band: Z-Image tops out at 1 MP, H3 starts at 4 MP", () => {
    expect(resolutionFor("16:9", 4, envelopeFor(Z_IMAGE))).toBe("1280x720");
    expect(resolutionFor("1:1", 1, resolutionEnvelope(H3_MODEL, H3_IMAGE))).toBe("2048x2048");
  });

  it("grows both sides to the side floor, keeping the shape", () => {
    const mage = resolutionEnvelope(MAGE, MAGE.limits.resolutions);
    expect(mage.minSide).toBe(512);
    expect(resolutionFor("4:1", 0, mage)).toBe("2048x512");
    expect(resolutionFor("21:9", 0, mage)).toBe("1216x512");
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
  it("disables a ratio the side range cannot draw", () => {
    const zImage = envelopeFor(Z_IMAGE);
    expect(zImage.ratios.find((ratio) => ratio.id === "21:9").available).toBe(false);
    expect(zImage.ratios.find((ratio) => ratio.id === "16:9").available).toBe(true);
    // SDXL's 896x1152 preset still counts as its 3:4.
    expect(envelopeFor(SDXL).ratios.find((ratio) => ratio.id === "3:4").available).toBe(true);
  });

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
    expect(resolutionFitsEnvelope("1536x2048", krea)).toBe(true);
    expect(resolutionFitsEnvelope("1536x2048", zImage)).toBe(false);
    expect(fitResolutionToEnvelope("1536x2048", zImage)).toBe("880x1184");
    // Past Krea's 2048 side: refit, never kept.
    expect(resolutionFitsEnvelope("2720x1536", krea)).toBe(false);
    // A preset of another model that this one can render is not rewritten.
    expect(resolutionFitsEnvelope("1216x832", krea)).toBe(true);
  });

  it("refits a carried-over size that is off the new model's grid", () => {
    const sdxl = envelopeFor(SDXL);
    expect(resolutionFitsEnvelope("880x1184", sdxl)).toBe(false);
    const [w, h] = fitResolutionToEnvelope("880x1184", sdxl).split("x").map(Number);
    expect(w % 64 + h % 64).toBe(0);
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
