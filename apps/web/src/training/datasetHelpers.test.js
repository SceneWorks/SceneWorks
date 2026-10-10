import { describe, expect, it } from "vitest";

import { summarize } from "../validation/issues.js";
import {
  appendReferences,
  datasetPayload,
  datasetReferenceAssets,
  datasetReferenceCap,
  datasetSaveValidation,
  editPairDatasetIssues,
  referenceDraftsDiffer,
  referenceDraftsFromDataset,
  selectionAfterDuplicateRemoval,
  trainingTargetReferenceCap,
} from "./datasetHelpers.js";

it("preserves prepared bundle extras and stable item ids across dataset saves", () => {
  const activeDataset = {
    id: "ds1",
    items: [{
      id: "item_1",
      path: "images/item_1.png",
      displayName: "one",
      ltxPreparedBundlePath: "prepared/item_1.safetensors",
      ltxPreparedBundleSize: 1234,
      ltxPreparedBundleSha256: "a".repeat(64),
      futurePreparedMetadata: { revision: 2 },
      controlImagePath: "controls/item_1.png",
      caption: { text: "one", source: "manual", triggerWords: [] },
    }],
  };
  const selection = "dataset-item:ds1:item_1";
  const payload = datasetPayload({
    activeDataset,
    assetsById: new Map([[selection, {
      id: selection,
      datasetOwned: true,
      displayName: "one",
      file: { path: "training/datasets/ds1/images/item_1.png" },
    }]]),
    name: "Prepared",
    selectedAssetIds: [selection],
  });
  expect(payload.items[0]).toMatchObject({
    id: "item_1",
    ltxPreparedBundlePath: "prepared/item_1.safetensors",
    ltxPreparedBundleSize: 1234,
    ltxPreparedBundleSha256: "a".repeat(64),
    futurePreparedMetadata: { revision: 2 },
    controlImagePath: "controls/item_1.png",
  });
});

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
// is a silent requirement; an empty selection is valid so a Parquet import has a persisted
// destination. The one thing that earns a chip is a selected asset that went unavailable.
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

  it("allows a named empty dataset for Parquet import", () => {
    const issues = datasetSaveValidation({ ...whole, selectedAssetIds: [] }, { health: { disabledItems: 0 } });
    expect(kinds(issues, "assets")).toEqual([]);
    expect(summarize(issues).surfaced).toEqual([]);
    expect(summarize(issues).ready).toBe(true);
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

  // A stale health count must not prevent creating a genuinely empty Parquet destination.
  it("ignores unavailable counts when the selection is empty", () => {
    const issues = datasetSaveValidation({ ...whole, selectedAssetIds: [] }, { health: { disabledItems: 5 } });
    expect(issues).toEqual([]);
  });

  it("tolerates a missing health context", () => {
    expect(() => datasetSaveValidation(whole)).not.toThrow();
    expect(summarize(datasetSaveValidation(whole)).ready).toBe(true);
  });
});

// sc-24161: instruction-edit pairs — ordered references round-trip through the save payload, the
// cap is the model's own reference limit, and the training gate mirrors the API's refusals.
describe("edit-pair dataset helpers (sc-24161)", () => {
  const editTarget = {
    id: "qwen_image_2_1_edit_lora",
    baseModel: "qwen_image_2_1",
    limits: { maxReferenceImages: 10 },
    ui: { label: "Qwen Image 2.1 Edit LoRA" },
  };
  const t2iTarget = { id: "qwen_image_2_1_lora", baseModel: "qwen_image_2_1", limits: {}, ui: { label: "Qwen Image 2.1 LoRA" } };
  const qwenModel = { id: "qwen_image_2_1", limits: { maxReferenceAssets: 10 } };

  it("keeps reference order through draft seeding and the save payload", () => {
    const dataset = {
      id: "ds1",
      items: [
        {
          id: "item_0001",
          assetId: "target",
          path: "images/item_0001.png",
          caption: { text: "swap the sky", source: "manual", triggerWords: [] },
          references: [
            { assetId: "ref-c", path: "images/refs/item_0001_ref1.png" },
            { path: "images/refs/item_0001_ref2.png", displayName: "upload.png" },
            { assetId: "ref-a", path: "images/refs/item_0001_ref3.png" },
          ],
        },
      ],
    };
    const catalog = [{ id: "target" }, { id: "ref-a" }, { id: "ref-c" }];
    const drafts = referenceDraftsFromDataset(dataset, catalog);
    expect(drafts).toEqual({ target: ["ref-c", "dataset-ref:ds1:item_0001:1", "ref-a"] });
    const owned = datasetReferenceAssets(dataset, "p1", catalog);
    expect(owned).toHaveLength(1);
    expect(owned[0]).toMatchObject({
      id: "dataset-ref:ds1:item_0001:1",
      datasetOwned: true,
      file: { path: "training/datasets/ds1/images/refs/item_0001_ref2.png" },
    });

    const assetsById = new Map(
      [...catalog.map((asset) => ({ ...asset, type: "image", displayName: `${asset.id}.png` })), ...owned].map((asset) => [
        asset.id,
        asset,
      ]),
    );
    // Reorder: the dataset-owned reference first.
    const reordered = { target: ["dataset-ref:ds1:item_0001:1", "ref-a", "ref-c"] };
    const payload = datasetPayload({
      activeDataset: dataset,
      assetsById,
      name: "Edits",
      selectedAssetIds: ["target"],
      referenceDraftById: reordered,
    });
    expect(payload.items[0].references).toEqual([
      { path: "training/datasets/ds1/images/refs/item_0001_ref2.png", displayName: "upload.png" },
      { assetId: "ref-a", displayName: "ref-a.png" },
      { assetId: "ref-c", displayName: "ref-c.png" },
    ]);
    expect(referenceDraftsDiffer(reordered, drafts, ["target"])).toBe(true);
    expect(referenceDraftsDiffer(drafts, drafts, ["target"])).toBe(false);

    // No drafts → no `references` key at all (plain items keep their exact payload shape), and a
    // stale stored list never rides along through the forward-compatible extras.
    const plain = datasetPayload({ activeDataset: dataset, assetsById, name: "Edits", selectedAssetIds: ["target"] });
    expect(plain.items[0]).not.toHaveProperty("references");
  });

  it("takes the reference cap from the model and never exceeds the target contract", () => {
    expect(trainingTargetReferenceCap(editTarget, [qwenModel])).toBe(10);
    expect(trainingTargetReferenceCap(editTarget, [{ ...qwenModel, limits: { maxReferenceAssets: 4 } }])).toBe(4);
    // A model that declares nothing falls back to the target's own cap.
    expect(trainingTargetReferenceCap(editTarget, [])).toBe(10);
    expect(trainingTargetReferenceCap(t2iTarget, [qwenModel])).toBe(0);
    expect(datasetReferenceCap([t2iTarget, editTarget], [qwenModel])).toBe(10);
    expect(datasetReferenceCap([t2iTarget], [qwenModel])).toBe(0);
  });

  it("appends picks in order, de-duplicated, and stops at the cap", () => {
    expect(appendReferences(["a"], ["b", "a", "item", "c"], { cap: 3, itemId: "item" })).toEqual({
      next: ["a", "b", "c"],
      dropped: 0,
    });
    expect(appendReferences(["a", "b"], ["c", "d", "e"], { cap: 3 })).toEqual({ next: ["a", "b", "c"], dropped: 2 });
  });

  it("holds training with the API's reasons for a dataset that does not fit the target", () => {
    const item = (id, refCount, text = "do the edit") => ({
      id,
      displayName: `${id}.png`,
      caption: { text },
      references: Array.from({ length: refCount }, (_, index) => ({ path: `images/refs/${id}_${index}.png` })),
    });
    const messages = (dataset, target, cap) => editPairDatasetIssues(dataset, target, cap).map((entry) => entry.message);

    expect(messages({ items: [item("a", 2), item("b", 10)] }, editTarget, 10)).toEqual([]);
    expect(messages({ items: [item("a", 2)] }, t2iTarget, 0)[0]).toContain("rejects references");
    expect(messages({ items: [item("a", 1), item("b", 0)] }, editTarget, 10)[0]).toContain("needs a");
    expect(messages({ items: [item("a", 11)] }, editTarget, 10)[0]).toContain("limit of 10");
    expect(messages({ items: [item("a", 1, "  ")] }, editTarget, 10)[0]).toContain("instructions");
    expect(messages({ items: [item("a", 0)] }, t2iTarget, 0)).toEqual([]);
  });
});

describe("edit-pair control conflict (sc-24161 review)", () => {
  it("refuses an item that is both an edit pair and a control pair", () => {
    const target = { id: "edit", limits: { maxReferenceImages: 10 }, ui: { label: "Edit" } };
    const item = {
      id: "a",
      displayName: "a.png",
      caption: { text: "edit it" },
      references: [{ path: "images/refs/a_ref1.png" }],
    };
    expect(editPairDatasetIssues({ items: [item] }, target, 10)).toEqual([]);
    const messages = editPairDatasetIssues({ items: [{ ...item, controlImagePath: "controls/a.png" }] }, target, 10).map(
      (entry) => entry.message,
    );
    expect(messages).toHaveLength(1);
    expect(messages[0]).toContain("control inputs");
  });
});
