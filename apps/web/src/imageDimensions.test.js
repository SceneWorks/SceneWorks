import { describe, expect, it, vi } from "vitest";
import { probeImageDimensions } from "./imageDimensions.js";

describe("probeImageDimensions", () => {
  it("reports natural dimensions and cleanup suppresses stale loads", () => {
    const OriginalImage = globalThis.Image;
    const probes = [];
    globalThis.Image = class {
      constructor() {
        this.naturalWidth = 640;
        this.naturalHeight = 480;
        probes.push(this);
      }
    };
    try {
      const onDimensions = vi.fn();
      const cleanup = probeImageDimensions("/asset.png", onDimensions);
      probes[0].onload();
      expect(onDimensions).toHaveBeenCalledWith(640, 480);

      cleanup();
      expect(probes[0].src).toBe("");
      expect(probes[0].onload).toBeNull();
    } finally {
      globalThis.Image = OriginalImage;
    }
  });
});
