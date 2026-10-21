import { describe, expect, it } from "vitest";

import { promptBatchSize, resolveCfgInterval } from "./samplingControls.js";

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
