import { afterEach, describe, expect, it, vi } from "vitest";
import { blobToImage, loadImage, probeImageDimensions } from "./imageDimensions.js";

afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("image loading", () => {
  it("times out an unresponsive probe and clears the image request", async () => {
    vi.useFakeTimers();
    let image;
    vi.stubGlobal("Image", class { constructor() { image = this; } });
    const result = loadImage("/image", { timeoutMs: 10 });
    const rejected = expect(result).rejects.toThrow("timed out");
    await vi.advanceTimersByTimeAsync(10);
    await rejected;
    expect(image.src).toBe("");
    expect(image.onload).toBeNull();
  });
  it("releases blob URLs on decode error and suppresses stale probes", async () => {
    const images = [];
    vi.stubGlobal("Image", class { constructor() { images.push(this); } });
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL: () => "blob:test", revokeObjectURL });
    const result = blobToImage(new Blob(["x"]));
    images[0].onerror();
    await expect(result).rejects.toThrow("Could not decode image");
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:test");
    const callback = vi.fn();
    const cancel = probeImageDimensions("/image", callback);
    const stale = images[1].onload;
    cancel();
    stale();
    await Promise.resolve();
    expect(callback).not.toHaveBeenCalled();
  });
});
