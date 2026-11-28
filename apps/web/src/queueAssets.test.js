import { expect, it, vi, afterEach } from "vitest";
import { assetsApi } from "./api/assetsResources.js";
import { loadQueueAssets } from "./queueAssets.js";
afterEach(() => vi.restoreAllMocks());

it("refreshes embedded output snapshots when entering Queue after an asset was trashed", async () => {
  const stale = { id: "output", status: { trashed: false } };
  const current = { ...stale, status: { trashed: true } };
  const item = vi.spyOn(assetsApi, "item").mockResolvedValue(current);
  expect(await loadQueueAssets("project", "token", {}, [
    { projectId: "project", result: { assets: [stale] } },
  ])).toEqual([current]);
  expect(item).toHaveBeenCalledWith("project", "output", "token", { signal: undefined });
});

it("loads only Queue output IDs and reuses known records instead of loading the asset library", async () => {
  const source = { id: "unrelated-source" };
  const output = { id: "output" };
  const item = vi.spyOn(assetsApi, "item").mockResolvedValue({ id: "new-output" });
  const list = vi.spyOn(assetsApi, "listSummaryIncludingHidden");
  const jobs = [{ projectId: "project", result: { assetIds: ["output", "new-output"] } }];
  expect(await loadQueueAssets("project", "token", {}, jobs, [source, output]))
    .toEqual([output, { id: "new-output" }]);
  expect(item).toHaveBeenCalledTimes(1);
  expect(item).toHaveBeenCalledWith("project", "new-output", "token", { signal: undefined });
  expect(list).not.toHaveBeenCalled();
});

it("limits concurrent output loads and stops on page disposal", async () => {
  let active = 0, peak = 0;
  const controller = new AbortController();
  const item = vi.spyOn(assetsApi, "item").mockImplementation(async (_project, id) => {
    peak = Math.max(peak, ++active);
    await Promise.resolve();
    active -= 1;
    if (id === "3") controller.abort();
    return { id };
  });
  const jobs = [{ projectId: "project", result: { assetIds: Array.from({ length: 100 }, (_, i) => String(i)) } }];
  await expect(loadQueueAssets("project", "token", { signal: controller.signal }, jobs)).rejects.toMatchObject({ name: "AbortError" });
  expect(peak).toBe(4);
  expect(item).toHaveBeenCalledTimes(4);
});

it("resolves generation-set-only saved jobs without retaining unrelated library assets", async () => {
  const output = { id: "legacy-output", generationSetId: "legacy-set" };
  vi.spyOn(assetsApi, "listSummaryIncludingHidden").mockResolvedValue([
    { id: "unrelated", generationSetId: "another-set" }, output,
  ]);
  expect(await loadQueueAssets("project", "token", {}, [
    { projectId: "project", result: { generationSetId: "legacy-set" } },
  ])).toEqual([output]);
});

it("resolves output records from every project in the Queue", async () => {
  const item = vi.spyOn(assetsApi, "item").mockImplementation(async (projectId, id) => ({ projectId, id }));
  expect(await loadQueueAssets("project-a", "token", {}, [
    { projectId: "project-a", result: { assetIds: ["a"] } },
    { projectId: "project-b", result: { assetIds: ["b"] } },
  ])).toEqual([{ projectId: "project-a", id: "a" }, { projectId: "project-b", id: "b" }]);
  expect(item).toHaveBeenCalledTimes(2);
});
