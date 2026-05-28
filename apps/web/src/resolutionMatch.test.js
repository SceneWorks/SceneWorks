import { describe, expect, it } from "vitest";
import { parseResolution, pickClosestResolution } from "./resolutionMatch.js";

describe("parseResolution", () => {
  it("parses WxH strings", () => {
    expect(parseResolution("1024x576")).toEqual({ width: 1024, height: 576 });
  });

  it("returns null for malformed input", () => {
    expect(parseResolution("1024")).toBeNull();
    expect(parseResolution("axb")).toBeNull();
    expect(parseResolution("")).toBeNull();
    expect(parseResolution(null)).toBeNull();
    expect(parseResolution(undefined)).toBeNull();
  });

  it("rejects non-positive dimensions", () => {
    expect(parseResolution("0x720")).toBeNull();
  });
});

describe("pickClosestResolution", () => {
  const ltxOptions = ["768x512", "512x768", "640x640", "1024x1024", "1280x720", "720x1280"];

  it("returns null when options or source dims are missing/invalid", () => {
    expect(pickClosestResolution(1024, 1024, [])).toBeNull();
    expect(pickClosestResolution(1024, 1024, null)).toBeNull();
    expect(pickClosestResolution(0, 1024, ltxOptions)).toBeNull();
    expect(pickClosestResolution(1024, 0, ltxOptions)).toBeNull();
    expect(pickClosestResolution(NaN, 1024, ltxOptions)).toBeNull();
  });

  it("picks the square option for a square source", () => {
    expect(pickClosestResolution(1500, 1500, ltxOptions)).toBe("1024x1024");
    expect(pickClosestResolution(500, 500, ltxOptions)).toBe("640x640");
  });

  it("picks landscape 16:9 for a 1920x1080 source", () => {
    expect(pickClosestResolution(1920, 1080, ltxOptions)).toBe("1280x720");
  });

  it("picks portrait 9:16 for a 1080x1920 source", () => {
    expect(pickClosestResolution(1080, 1920, ltxOptions)).toBe("720x1280");
  });

  it("picks landscape 3:2 for a 3000x2000 source", () => {
    expect(pickClosestResolution(3000, 2000, ltxOptions)).toBe("768x512");
  });

  it("picks portrait 2:3 for a 2000x3000 source", () => {
    expect(pickClosestResolution(2000, 3000, ltxOptions)).toBe("512x768");
  });

  it("skips malformed options without throwing", () => {
    expect(pickClosestResolution(1500, 1500, ["bad", "1024x1024"])).toBe("1024x1024");
  });
});
