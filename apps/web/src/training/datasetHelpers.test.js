import { describe, expect, it } from "vitest";

import { summarize } from "../validation/issues.js";
import { datasetSaveValidation, selectionAfterDuplicateRemoval } from "./datasetHelpers.js";

describe("selectionAfterDuplicateRemoval (sc-6539 one-tap dedupe mapping)", () => {
  // Mix of catalog-backed items (selection key = assetId) and a dataset-owned item (no assetId, so the
  // key is the synthesized `dataset-item:<dsid>:<itemid>` — the case most at risk of a mapping miss).
  const dataset = {
    id: "ds1",
    items: [
      { id: "item_0001", assetId: "asset-a" },
      { id: "item_0002", assetId: "asset-b" },
      { id: "item_0003" },
    ],
  };
  const currentSelection = ["asset-a", "asset-b", "dataset-item:ds1:item_0003"];

  it("drops a catalog-backed duplicate by its asset-id key, keeping the rest", () => {
    const { nextSelection, removedCount } = selectionAfterDuplicateRemoval({
      dataset,
      currentSelection,
      removeIds: ["item_0002"],
    });
    expect(removedCount).toBe(1);
    expect(nextSelection).toEqual(["asset-a", "dataset-item:ds1:item_0003"]);
  });

  it("drops a dataset-owned (non-catalog) duplicate by its synthesized selection key", () => {
    const { nextSelection, removedCount } = selectionAfterDuplicateRemoval({
      dataset,
      currentSelection,
      removeIds: ["item_0003"],
    });
    expect(removedCount).toBe(1);
    expect(nextSelection).toEqual(["asset-a", "asset-b"]);
  });

  it("removes every planned duplicate at once", () => {
    const { nextSelection, removedCount } = selectionAfterDuplicateRemoval({
      dataset,
      currentSelection,
      removeIds: ["item_0001", "item_0002"],
    });
    expect(removedCount).toBe(2);
    expect(nextSelection).toEqual(["dataset-item:ds1:item_0003"]);
  });

  it("is a no-op when the planned ids are no longer in the dataset (stale report)", () => {
    const { nextSelection, removedCount } = selectionAfterDuplicateRemoval({
      dataset,
      currentSelection,
      removeIds: ["item_9999"],
    });
    expect(removedCount).toBe(0);
    expect(nextSelection).toEqual(currentSelection);
  });

  it("handles empty / missing inputs without throwing", () => {
    expect(selectionAfterDuplicateRemoval({})).toEqual({ nextSelection: [], removedCount: 0 });
    expect(
      selectionAfterDuplicateRemoval({ dataset, currentSelection, removeIds: [] }),
    ).toEqual({ nextSelection: currentSelection, removedCount: 0 });
  });
});

// The dataset-save gate in the app-wide vocabulary (epic 10644, sc-10648). A missing name
// or empty selection is a silent requirement; the one thing that earns a chip is a
// selection holding assets that went unavailable after they were picked.
describe("datasetSaveValidation", () => {
  const whole = { name: "Kelsie", selectedAssetIds: ["a", "b"] };
  const kinds = (issues, field) => issues.filter((entry) => entry.field === field).map((entry) => entry.kind);

  it("passes a named, populated, healthy selection", () => {
    const summary = summarize(datasetSaveValidation(whole, { health: { disabledItems: 0 } }));
    expect(summary.ready).toBe(true);
    expect(summary.surfaced).toEqual([]);
  });

  it("requires a name, silently", () => {
    const issues = datasetSaveValidation({ ...whole, name: "  " }, { health: { disabledItems: 0 } });
    expect(kinds(issues, "datasetName")).toEqual(["requirement"]);
    expect(summarize(issues).surfaced).toEqual([]);
    expect(summarize(issues).ready).toBe(false);
  });

  it("requires at least one asset, silently", () => {
    const issues = datasetSaveValidation({ ...whole, selectedAssetIds: [] }, { health: { disabledItems: 0 } });
    expect(kinds(issues, "assets")).toEqual(["requirement"]);
    expect(summarize(issues).surfaced).toEqual([]);
  });

  // The one improvement this migration buys: a dead Save that used to say nothing (the
  // health dot read "Add image assets", which is wrong when the set is full of bad ones).
  it("raises a surfaced error when the selection holds unavailable assets", () => {
    const summary = summarize(datasetSaveValidation(whole, { health: { disabledItems: 2 } }));
    expect(summary.ready).toBe(false);
    expect(summary.surfaced).toHaveLength(1);
    expect(summary.surfaced[0].kind).toBe("error");
    expect(summary.surfaced[0].message).toContain("2 unavailable images");
  });

  it("singularizes the unavailable-asset message", () => {
    const summary = summarize(datasetSaveValidation(whole, { health: { disabledItems: 1 } }));
    expect(summary.surfaced[0].message).toContain("1 unavailable image ");
    expect(summary.surfaced[0].message).toContain("it has been");
  });

  // An empty selection is a requirement, not the disabledItems error — order matters, or a
  // brand-new dataset would nag about "unavailable images" it doesn't have.
  it("prefers the empty-selection requirement over the unavailable error", () => {
    const issues = datasetSaveValidation({ ...whole, selectedAssetIds: [] }, { health: { disabledItems: 5 } });
    expect(issues).toHaveLength(1);
    expect(issues[0].kind).toBe("requirement");
  });

  it("tolerates a missing health context", () => {
    expect(() => datasetSaveValidation(whole)).not.toThrow();
    expect(summarize(datasetSaveValidation(whole)).ready).toBe(true);
  });
});
