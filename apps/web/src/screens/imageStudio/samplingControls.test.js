import { describe, expect, it } from "vitest";

import {
  h3ImageSamplerDefaults,
  h3ImageSamplerFromRecipe,
  promptBatchSize,
  resolveCfgInterval,
  resolveH3ImageSampler,
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

  it("restores a recorded block, with the engine's defaults for omitted LCM fields", () => {
    expect(h3ImageSamplerFromRecipe({ sampler: "euler" }).sampler).toBe("euler");
    expect(h3ImageSamplerFromRecipe({ sampler: "lcm", sNoise: 1.3 })).toEqual({
      sampler: "lcm", sNoise: 1.3, sNoiseEnd: 1.3, noiseClipStd: 0,
    });
  });

  it("replays a job or preset recorded with no block as Euler, which is how it ran", () => {
    for (const recorded of [undefined, null, {}]) {
      expect(h3ImageSamplerFromRecipe(recorded)).toEqual({ sampler: "euler", sNoise: 1.1, sNoiseEnd: 0.9, noiseClipStd: 0 });
    }
    expect(resolveH3ImageSampler({ model: h3, options: h3ImageSamplerFromRecipe(undefined) }).block).toEqual({ sampler: "euler" });
  });
});
