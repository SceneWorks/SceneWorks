import { describe, expect, it } from "vitest";

import {
  H3_IMAGE_LCM_FIELDS,
  h3ImageSamplerDefaults,
  h3ImageSamplerFromRecipe,
  pagDefaults,
  pagFromRecipe,
  promptBatchSize,
  resolveCfgInterval,
  resolveH3ImageSampler,
  resolvePag,
  supportsPag,
} from "./samplingControls.js";

// sc-25681: Image Studio's Iris-only sampling controls.
describe("samplingControls", () => {
  const iris = { defaults: { guidanceScale: 3.0 }, image: { supportsCfgInterval: true, maxPromptBatchImages: 8 } };
  const plain = { defaults: { guidanceScale: 3.0 } };
  const bad = { interval: null, error: "Need 0 ≤ start < end ≤ 1." };

  it.each([
    // [text, guidance, result]
    ["", "", { interval: null, error: null }],
    ["0, 1", "", { interval: null, error: null }],
    ["0.1, 0.9", "", { interval: [0.1, 0.9], error: null }],
    ["0,0.6", "4", { interval: [0, 0.6], error: null }],
    ["0.1, 0.9", "1", { interval: null, error: null }],
    ["0.9, 0.1", "", bad],
    ["-0.1, 0.5", "", bad],
    ["0.2, 1.5", "", bad],
    ["0.2", "", bad],
  ])("cfg interval %j at guidance %j", (text, guidanceOverride, result) => {
    expect(resolveCfgInterval({ model: iris, text, guidanceOverride })).toEqual(result);
  });

  it("never resolves an interval for a model that does not offer one", () => {
    expect(resolveCfgInterval({ model: plain, text: "0.1, 0.9", guidanceOverride: "" })).toEqual({
      interval: null,
      error: null,
    });
  });

  // Mutation that reds this: divide by 1 instead of `count`.
  it("packs floor(cap / count) prompts per native job, 1 without a cap", () => {
    expect(promptBatchSize(iris, 1)).toBe(8);
    expect(promptBatchSize(iris, 4)).toBe(2);
    expect(promptBatchSize(iris, 8)).toBe(1);
    expect(promptBatchSize(plain, 1)).toBe(1);
  });
});

// SW-7106: MiniMax-H3-Image's sampler control.
describe("MiniMax-H3-Image sampler", () => {
  const h3 = { image: { supportsH3ImageSampler: true } };
  const lcm = { ...h3ImageSamplerDefaults(), sampler: "lcm" };

  it("defaults to LCM with noise scale 1.1 / 0.9 and clip 0", () => {
    expect(h3ImageSamplerDefaults()).toEqual({ sampler: "lcm", sNoise: 1.1, sNoiseEnd: 0.9, noiseClipStd: 0 });
  });

  it("sends the default LCM block explicitly, Euler as {sampler: euler}, and no block without the control", () => {
    expect(resolveH3ImageSampler({ model: h3, options: h3ImageSamplerDefaults() })).toEqual({
      block: { sampler: "lcm", sNoise: 1.1, sNoiseEnd: 0.9, noiseClipStd: 0 },
      error: null,
    });
    expect(resolveH3ImageSampler({ model: h3, options: { ...h3ImageSamplerDefaults(), sampler: "euler" } })).toEqual({ block: { sampler: "euler" }, error: null });
    expect(resolveH3ImageSampler({ model: { image: {} }, options: lcm })).toEqual({ block: null, error: null });
  });

  it("sends LCM's numeric fields and names the first out-of-range one", () => {
    expect(resolveH3ImageSampler({ model: h3, options: { ...lcm, sNoise: "1.25", noiseClipStd: "2" } }).block).toEqual({
      sampler: "lcm", sNoise: 1.25, sNoiseEnd: 0.9, noiseClipStd: 2,
    });
    for (const [patch, label] of [
      [{ sNoise: "0" }, "Noise scale start must be between 0.01 and 2."],
      [{ sNoise: "" }, "Noise scale start must be between 0.01 and 2."],
      [{ sNoiseEnd: "-0.1" }, "Noise scale end must be between 0 and 2."],
      [{ noiseClipStd: "4.5" }, "Noise clip (std, 0 = off) must be between 0 and 4."],
    ]) {
      expect(resolveH3ImageSampler({ model: h3, options: { ...lcm, ...patch } })).toEqual({ block: null, error: label });
    }
  });

  it("accepts a bound that came back through the worker's f32 record, and still refuses a real overshoot", () => {
    // The worker records each scale as f32: a UI 0.01 replays as 0.009999999776482582.
    const f32 = (value) => Math.fround(value);
    expect(f32(0.01)).toBe(0.009999999776482582);
    const replayed = { ...lcm, sNoise: f32(0.01), sNoiseEnd: f32(2), noiseClipStd: f32(4) };
    expect(resolveH3ImageSampler({ model: h3, options: replayed })).toEqual({
      block: { sampler: "lcm", sNoise: f32(0.01), sNoiseEnd: 2, noiseClipStd: 4 },
      error: null,
    });
    // A max bound that is not exact in f32 still admits its own f32 round trip.
    expect(resolveH3ImageSampler({ model: h3, options: { ...lcm, sNoiseEnd: f32(1.9), sNoise: f32(0.11) } }).error).toBeNull();
    for (const [patch, label] of [
      [{ sNoise: 0.0099 }, "Noise scale start must be between 0.01 and 2."],
      [{ sNoise: 2.0001 }, "Noise scale start must be between 0.01 and 2."],
      [{ sNoiseEnd: -0.0001 }, "Noise scale end must be between 0 and 2."],
      [{ noiseClipStd: 4.0001 }, "Noise clip (std, 0 = off) must be between 0 and 4."],
    ]) {
      expect(resolveH3ImageSampler({ model: h3, options: { ...lcm, ...patch } })).toEqual({ block: null, error: label });
    }
  });

  it("restores a recorded block, with the engine's defaults for omitted LCM fields", () => {
    expect(h3ImageSamplerFromRecipe({ sampler: "euler" }).sampler).toBe("euler");
    // A complete block restores as recorded.
    expect(h3ImageSamplerFromRecipe({ sampler: "lcm", sNoise: 1.4, sNoiseEnd: 0.6, noiseClipStd: 2.5 })).toEqual({
      sampler: "lcm", sNoise: 1.4, sNoiseEnd: 0.6, noiseClipStd: 2.5,
    });
    // A partial `{sampler: "lcm"}` ran 1.1 -> 0.9 and replays the same.
    expect(h3ImageSamplerFromRecipe({ sampler: "lcm" })).toEqual({
      sampler: "lcm", sNoise: 1.1, sNoiseEnd: 0.9, noiseClipStd: 0,
    });
    // A lone sNoise holds constant, the worker's rule.
    expect(h3ImageSamplerFromRecipe({ sampler: "lcm", sNoise: 1.3 })).toEqual({
      sampler: "lcm", sNoise: 1.3, sNoiseEnd: 1.3, noiseClipStd: 0,
    });
    expect(h3ImageSamplerFromRecipe({ sampler: "lcm", sNoiseEnd: 0.5 })).toEqual({
      sampler: "lcm", sNoise: 1.1, sNoiseEnd: 0.5, noiseClipStd: 0,
    });
  });

  it("restores f32-recorded scales as their decimal form, inside each input's native bounds", () => {
    const f32 = (value) => Math.fround(value);
    const restored = h3ImageSamplerFromRecipe({ sampler: "lcm", sNoise: f32(0.01), sNoiseEnd: f32(1.9), noiseClipStd: f32(0.3) });
    expect(restored).toEqual({ sampler: "lcm", sNoise: 0.01, sNoiseEnd: 1.9, noiseClipStd: 0.3 });
    // The value the number input shows satisfies its own `min`/`max`, so native form validation
    // lets Generate submit.
    for (const { key, min, max } of H3_IMAGE_LCM_FIELDS) {
      expect(restored[key] >= min && restored[key] <= max).toBe(true);
    }
    // A value that differs at f32 precision keeps its difference and is still refused.
    expect(h3ImageSamplerFromRecipe({ sampler: "lcm", sNoise: 0.0099 }).sNoise).toBe(0.0099);
  });

  it("replays a job or preset recorded with no block as Euler, which is how it ran", () => {
    for (const recorded of [undefined, null, {}]) {
      expect(h3ImageSamplerFromRecipe(recorded)).toEqual({ sampler: "euler", sNoise: 1.1, sNoiseEnd: 0.9, noiseClipStd: 0 });
    }
    expect(resolveH3ImageSampler({ model: h3, options: h3ImageSamplerFromRecipe(undefined) }).block).toEqual({ sampler: "euler" });
  });
});

// SW-7181: Capability-driven Perturbed-Attention Guidance (PAG) helpers.
describe("PAG sampling controls", () => {
  const pagModel = {
    id: "krea_2_raw",
    pag: {
      numBlocks: 28,
      blockDomain: "single_stream_blocks",
      defaultLayers: [14],
      defaultStart: 0.1,
      defaultStop: 0.75,
      recommendedScale: 1.5,
      maxScale: 15,
    },
  };

  it("gates capability solely on model.pag.numBlocks > 0", () => {
    expect(supportsPag(pagModel)).toBe(true);
    expect(supportsPag({ id: "sdxl" })).toBe(false);
    expect(supportsPag({ id: "bad", pag: { numBlocks: 0 } })).toBe(false);
    expect(supportsPag(null)).toBe(false);
  });

  it("omits advanced.pag when unsupported, blank, or zero", () => {
    expect(resolvePag({ model: { id: "sdxl" }, options: { scale: "1.5" } })).toEqual({ block: null, error: null });
    for (const scale of ["", "   ", null, undefined, 0, "0", "0.0"]) {
      expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale } })).toEqual({
        block: null,
        error: null,
      });
    }
  });

  it("emits clean blocks with optional overrides omitted when blank or zero adaptiveScale", () => {
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5" } })).toEqual({
      block: { scale: 1.5 },
      error: null,
    });
    expect(
      resolvePag({
        model: pagModel,
        options: { scale: "2", layers: "0, 14 27", start: "0", stop: "0.9", adaptiveScale: "0.25" },
      }),
    ).toEqual({
      block: { scale: 2, layers: [0, 14, 27], start: 0, stop: 0.9, adaptiveScale: 0.25 },
      error: null,
    });
    expect(
      resolvePag({
        model: pagModel,
        options: { scale: "2", layers: "", start: "", stop: "", adaptiveScale: "0" },
      }),
    ).toEqual({
      block: { scale: 2 },
      error: null,
    });
  });

  it("refuses out-of-range scales, invalid/duplicate layers, and inverted or out-of-bounds windows", () => {
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "-0.1" } }).error).toContain("between 0 and 15");
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "15.5" } }).error).toContain("between 0 and 15");
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "abc" } }).error).toContain("between 0 and 15");

    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5", layers: "28" } }).error).toContain("0 and 27");
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5", layers: "-1" } }).error).toContain("integers");
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5", layers: "1.5" } }).error).toContain("integers");
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5", layers: "4, 4" } }).error).toContain("duplicated");
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5", layers: ", ," } }).error).toContain("integers");

    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5", start: "-0.1" } }).error).toContain("between 0 and 1");
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5", stop: "1.1" } }).error).toContain("between 0 and 1");
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5", start: "0.8", stop: "0.2" } }).error).toContain("≤ stop");
    // Single-bound override compared against model defaults (defaultStart=0.1, defaultStop=0.75).
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5", start: "0.8" } }).error).toContain("≤ stop (0.75)");
    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5", stop: "0.05" } }).error).toContain("≤ stop (0.05)");

    expect(resolvePag({ model: pagModel, options: { ...pagDefaults(), scale: "1.5", adaptiveScale: "-0.1" } }).error).toContain("0 or greater");
  });

  it("rehydrates recorded PAG blocks cleanly without reading execution reports", () => {
    expect(pagFromRecipe(undefined)).toEqual(pagDefaults());
    expect(pagFromRecipe(null)).toEqual(pagDefaults());
    expect(pagFromRecipe({ scale: 0, layers: [14] })).toEqual(pagDefaults());
    expect(
      pagFromRecipe({
        scale: Math.fround(1.5),
        layers: [2, 5],
        start: 0,
        stop: Math.fround(0.75),
        adaptive_scale: Math.fround(0.3),
      }),
    ).toEqual({
      scale: "1.5",
      layers: "2, 5",
      start: "0",
      stop: "0.75",
      adaptiveScale: "0.3",
    });
    expect(pagFromRecipe({ scale: 1.2, adaptiveScale: 0 })).toEqual({
      scale: "1.2",
      layers: "",
      start: "",
      stop: "",
      adaptiveScale: "",
    });
  });
});
