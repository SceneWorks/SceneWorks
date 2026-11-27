import { describe, expect, it, vi } from "vitest";
import { ensureFullAsset, isSummaryAsset, mergeAssetChanges, normalizeAssetChanges } from "./assetCatalog.js";

const asset = (id, createdAt, extra = {}) => ({ id, createdAt, ...extra });

describe("normalizeAssetChanges", () => {
  it("treats a bare array as a full replacement with no cursor", () => {
    expect(normalizeAssetChanges([asset("a", "1")])).toEqual({ reset: true, cursor: null, assets: [asset("a", "1")], removedIds: [] });
  });
  it("reads the incremental envelope and fails closed to a reset", () => {
    expect(normalizeAssetChanges({ cursor: "e.4", reset: false, assets: [asset("a", "1")], removedIds: ["b"] }))
      .toEqual({ reset: false, cursor: "e.4", assets: [asset("a", "1")], removedIds: ["b"] });
    expect(normalizeAssetChanges({ cursor: "", assets: "nope" })).toEqual({ reset: true, cursor: null, assets: [], removedIds: [] });
    expect(normalizeAssetChanges(null).reset).toBe(true);
  });
});

describe("mergeAssetChanges", () => {
  const current = [asset("c", "2026-03"), asset("b", "2026-02"), asset("a", "2026-01")];

  it("returns the same catalog when nothing changed", () => {
    expect(mergeAssetChanges(current, { assets: [], removedIds: [] })).toBe(current);
  });

  it("replaces changed assets in place, drops removed ids and keeps untouched identities", () => {
    const changedB = asset("b", "2026-02", { status: { favorite: true } });
    const merged = mergeAssetChanges(current, { assets: [changedB], removedIds: ["a", "unknown"] });
    expect(merged.map((item) => item.id)).toEqual(["c", "b"]);
    expect(merged[1]).toBe(changedB);
    expect(merged[0]).toBe(current[0]);
  });

  it("merges new assets in newest-first order, ahead of equal timestamps", () => {
    const merged = mergeAssetChanges(current, {
      assets: [asset("d", "2026-04"), asset("between", "2026-02-15"), asset("tie", "2026-01")],
      removedIds: [],
    });
    expect(merged.map((item) => item.id)).toEqual(["d", "c", "between", "b", "tie", "a"]);
  });

  it("re-adds an asset that came back into the filter", () => {
    const merged = mergeAssetChanges([asset("c", "2026-03")], { assets: [asset("a", "2026-01")], removedIds: [] });
    expect(merged.map((item) => item.id)).toEqual(["c", "a"]);
  });
});

describe("ensureFullAsset", () => {
  it("passes full assets through without loading", async () => {
    const load = vi.fn();
    const full = asset("a", "1", { recipe: { loras: [] } });
    expect(await ensureFullAsset(full, load)).toBe(full);
    expect(load).not.toHaveBeenCalled();
  });

  it("loads the full envelope for a summary item", async () => {
    const summary = asset("a", "1", { projectId: "p", listView: "summary" });
    const full = asset("a", "1", { projectId: "p", recipe: { loras: [{ id: "l" }] } });
    const load = vi.fn().mockResolvedValue(full);
    expect(isSummaryAsset(summary)).toBe(true);
    expect(await ensureFullAsset(summary, load)).toBe(full);
    expect(load).toHaveBeenCalledWith("p", "a");
    expect(isSummaryAsset(full)).toBe(false);
  });

  it("refuses to fall back to the partial summary", async () => {
    await expect(ensureFullAsset({ id: "a", projectId: "p", listView: "summary" }, async () => null))
      .rejects.toThrow("no longer available");
  });
});
