import { describe, expect, it } from "vitest";
import { describeResolution, nearestAspectName, parseSize, resolutionSummary } from "./aspect.js";

describe("parseSize", () => {
  it("parses a WxH string and rejects anything else", () => {
    expect(parseSize("1344x768")).toEqual({ width: 1344, height: 768 });
    expect(parseSize("1024X1024")).toEqual({ width: 1024, height: 1024 });
    expect(parseSize("1024")).toBeNull();
    expect(parseSize("0x512")).toBeNull();
    expect(parseSize("")).toBeNull();
    expect(parseSize(null)).toBeNull();
  });
});

describe("nearestAspectName", () => {
  it("names the standard generation buckets", () => {
    // The reason this isn't a gcd reduction: these buckets are latent-grid-aligned, so
    // the exact ratios are 7:4, 7:9 and 19:13 — correct and useless as a shape cue.
    expect(nearestAspectName(1024, 1024)).toBe("1:1");
    expect(nearestAspectName(1344, 768)).toBe("16:9");
    expect(nearestAspectName(768, 1344)).toBe("9:16");
    expect(nearestAspectName(1216, 832)).toBe("3:2");
    expect(nearestAspectName(1280, 720)).toBe("16:9");
    expect(nearestAspectName(720, 1280)).toBe("9:16");
    expect(nearestAspectName(854, 480)).toBe("16:9");
  });

  it("picks the CLOSEST name, not the conventional one, for the near-square buckets", () => {
    // 1152×896 is 1.2857 — nearer 5:4 (1.25) than 4:3 (1.333). The exact pixels are
    // always shown beside the label, so the closer name is the honest one.
    expect(nearestAspectName(1152, 896)).toBe("5:4");
    expect(nearestAspectName(896, 1152)).toBe("4:5");
  });

  it("handles wide and tall extremes", () => {
    expect(nearestAspectName(2560, 1080)).toBe("21:9");
    expect(nearestAspectName(1080, 2560)).toBe("9:21");
  });
});

describe("describeResolution / resolutionSummary", () => {
  it("pairs the shape name with the exact pixels", () => {
    expect(describeResolution("1344x768")).toEqual({ label: "16:9", sub: "1344×768" });
    expect(resolutionSummary("1344x768")).toBe("16:9 · 1344×768");
  });

  it("degrades to the raw string rather than inventing a shape", () => {
    expect(describeResolution("weird")).toEqual({ label: "weird", sub: "" });
    expect(resolutionSummary("weird")).toBe("weird");
    expect(describeResolution(null)).toEqual({ label: "", sub: "" });
  });
});
