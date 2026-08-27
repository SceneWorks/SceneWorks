import { describe, expect, it, vi } from "vitest";

import { createLatestMaskAssetLoader } from "./maskAssetLoader.js";

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

describe("createLatestMaskAssetLoader", () => {
  it("aborts a superseded asset fetch and commits only the latest mask", async () => {
    const firstFetch = deferred();
    const secondFetch = deferred();
    const fetchImpl = vi.fn()
      .mockReturnValueOnce(firstFetch.promise)
      .mockReturnValueOnce(secondFetch.promise);
    const blobToImage = vi.fn(async (blob) => ({ image: { id: blob.id }, objectUrl: `blob:${blob.id}` }));
    const loader = createLatestMaskAssetLoader({
      fetchImpl,
      assetUrlFor: (asset) => asset.url,
      blobToImage,
    });
    const revokeObjectURL = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const committed = [];

    const firstGeneration = loader.invalidate();
    const firstLoad = loader.load({ url: "old" }, firstGeneration, (image) => committed.push(image.id));
    const firstSignal = fetchImpl.mock.calls[0][1].signal;

    const secondGeneration = loader.invalidate();
    expect(firstSignal.aborted).toBe(true);
    const secondLoad = loader.load({ url: "new" }, secondGeneration, (image) => committed.push(image.id));

    firstFetch.resolve({ ok: true, blob: async () => ({ id: "old" }) });
    secondFetch.resolve({ ok: true, blob: async () => ({ id: "new" }) });

    await expect(firstLoad).resolves.toBe(false);
    await expect(secondLoad).resolves.toBe(true);
    expect(committed).toEqual(["new"]);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:old");
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:new");
    revokeObjectURL.mockRestore();
  });
});
