import { describe, expect, it } from "vitest";

import {
  executionRepresentationLabel,
  formatBytes,
  formatMs,
  formatPercent,
  quantLabel,
  sourceCodecLabel,
  weightFactsLabel,
} from "./formatting.js";

describe("formatMs", () => {
  it("reads sub-second in ms, then seconds, then minutes", () => {
    expect(formatMs(820)).toBe("820 ms");
    expect(formatMs(9400)).toBe("9.4s");
    expect(formatMs(65000)).toBe("1m 05s");
  });
  it("returns a dash for null/undefined/non-finite", () => {
    expect(formatMs(null)).toBe("—");
    expect(formatMs(undefined)).toBe("—");
    expect(formatMs(Number.NaN)).toBe("—");
  });
});

describe("formatBytes", () => {
  it("scales to binary units", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1024)).toBe("1.0 KiB");
    expect(formatBytes(12_884_901_888)).toBe("12.0 GiB");
  });
  it("returns a dash for null", () => {
    expect(formatBytes(null)).toBe("—");
  });
});

describe("formatPercent", () => {
  it("rounds a 0..100 value", () => {
    expect(formatPercent(71.5)).toBe("72%");
    expect(formatPercent(0)).toBe("0%");
    expect(formatPercent(null)).toBe("—");
  });
});

describe("quantLabel", () => {
  it("passes through a label or falls back to a dash", () => {
    expect(quantLabel("q8")).toBe("q8");
    expect(quantLabel("int8-convrot")).toBe("int8-convrot");
    expect(quantLabel("")).toBe("—");
    expect(quantLabel(null)).toBe("—");
  });
});

// sc-21484, epic 11037: the three facts stay three facts on the way to the screen.
describe("source codec vs execution representation", () => {
  it("renders the engine's execution labels and never invents one", () => {
    expect(executionRepresentationLabel("native-packed")).toBe("native (packed)");
    expect(executionRepresentationLabel("dense-fallback")).toBe("dense fallback");
    // Unmeasured is its own answer. It must not read as dense, and must not read as native.
    for (const absent of [null, undefined, "", "   "]) {
      expect(executionRepresentationLabel(absent)).toBe("not measured");
    }
    const unmeasured = executionRepresentationLabel(null);
    expect(unmeasured).not.toMatch(/dense/i);
    expect(unmeasured).not.toMatch(/native/i);
    // An unknown label from a newer engine passes through rather than being mapped to a guess.
    expect(executionRepresentationLabel("some-future-representation")).toBe(
      "some-future-representation",
    );
  });

  it("passes the codec id through without shortening it to the tier spelling", () => {
    expect(sourceCodecLabel("nvfp4-v1")).toBe("nvfp4-v1");
    expect(sourceCodecLabel("nvfp4-v1")).not.toBe("nvfp4");
    expect(sourceCodecLabel("int8-per-row-v1")).toBe("int8-per-row-v1");
    expect(sourceCodecLabel(null)).toBe("—");
    expect(sourceCodecLabel("")).toBe("—");
  });

  it("keeps the stored codec and the executed representation visibly distinct", () => {
    // A pre-Blackwell / sm_100 / CPU host running an NVFP4-stored checkpoint: the row says BOTH.
    expect(
      weightFactsLabel({
        quantLabel: "nvfp4",
        sourceCodec: "nvfp4-v1",
        executionRepresentation: "dense-fallback",
      }),
    ).toBe("nvfp4-v1 · dense fallback");
    // sm_120, measured native.
    expect(
      weightFactsLabel({ sourceCodec: "nvfp4-v1", executionRepresentation: "native-packed" }),
    ).toBe("nvfp4-v1 · native (packed)");
    // Known codec, no receipt: the codec must NOT stand in for the execution.
    const unmeasured = weightFactsLabel({ quantLabel: "nvfp4", sourceCodec: "nvfp4-v1" });
    expect(unmeasured).toBe("nvfp4-v1 · not measured");
    expect(unmeasured).not.toMatch(/native/i);
    // No classification at all: a dash, never the requested tier echoed back.
    expect(weightFactsLabel({ quantLabel: "q4" })).toBe("—");
    expect(weightFactsLabel(undefined)).toBe("—");
  });
});
