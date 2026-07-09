import { describe, expect, it } from "vitest";

import { formatBytes, formatMs, formatPercent, quantLabel } from "./formatting.js";

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
