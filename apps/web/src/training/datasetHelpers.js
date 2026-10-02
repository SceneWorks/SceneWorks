// Pure dataset helpers for the Training Studio (sc-4199). Extracted verbatim
// from TrainingStudio.jsx: dataset summaries, selection-key derivation, the
// owned-asset/asset-id normalizers, dataset-health math, and the save payload
// builder. No React, no app state — just data shaping over dataset records.

import { maxReferencesForModel } from "../imageReferenceLimits.js";
import { issue } from "../validation/issues.js";

export function imageAssetName(asset) {
  const path = asset?.file?.path ?? asset?.path ?? asset?.displayName ?? asset?.id ?? "asset";
  return String(path).replaceAll("\\", "/").split("/").pop() || "asset";
}

export function datasetItemCount(dataset) {
  const value = Number(dataset.itemCount ?? dataset.items?.length ?? 0);
  return Number.isFinite(value) ? value : 0;
}

export function captionText(item) {
  return String(item?.caption?.text ?? "").trim();
}

export function datasetItemSelectionKey(dataset, item, index = 0) {
  return item?.assetId || `dataset-item:${dataset?.id ?? "draft"}:${item?.id ?? index}`;
}

export function datasetItemProjectPath(dataset, item) {
  const path = String(item?.path ?? "").replaceAll("\\", "/");
  if (!dataset?.id || !path) {
    return "";
  }
  return `training/datasets/${dataset.id}/${path}`;
}

// Caption edit state keyed by selection id (sc-2025): the single source of
// truth for the unified caption cards, seeded from the saved dataset items and
// updated as the user edits or imports captions.
export function captionDraftsFromDataset(dataset) {
  const map = {};
  (dataset?.items ?? []).forEach((item, index) => {
    map[datasetItemSelectionKey(dataset, item, index)] = {
      text: item.caption?.text ?? "",
      source: item.caption?.source ?? "manual",
    };
  });
  return map;
}

export function datasetOwnedAssets(dataset, projectId, catalogAssets = []) {
  const catalogIds = new Set(catalogAssets.map((asset) => asset.id));
  return (dataset?.items ?? [])
    .map((item, index) => {
      if (item.assetId && catalogIds.has(item.assetId)) {
        return null;
      }
      const path = datasetItemProjectPath(dataset, item);
      if (!path) {
        return null;
      }
      const id = datasetItemSelectionKey(dataset, item, index);
      return {
        id,
        assetId: item.assetId ?? null,
        datasetOwned: true,
        projectId,
        type: "image",
        displayName: item.displayName ?? imageAssetName(item),
        file: {
          path,
          mimeType: `image/${String(path).split(".").pop() || "png"}`,
          width: item.width ?? null,
          height: item.height ?? null,
        },
      };
    })
    .filter(Boolean);
}

export function normalizeDatasetAssetIds(dataset, catalogAssets = []) {
  const catalogIds = new Set(catalogAssets.map((asset) => asset.id));
  return (dataset?.items ?? [])
    .map((item, index) => {
      if (item.assetId && catalogIds.has(item.assetId)) {
        return item.assetId;
      }
      return datasetItemSelectionKey(dataset, item, index);
    })
    .filter(Boolean);
}

// Map the readiness report's duplicate item ids (server item ids) to the selection keys to drop, and
// return the dataset's selection with those removed (sc-6539 one-tap dedupe). Pure so the mutating
// apply path is testable without rendering the studio: callers pass the freshly-saved dataset (for a
// stable id→key mapping) and its current selection. `removedCount` is the number actually matched —
// 0 when the plan's ids are no longer present, so the caller can skip a no-op write.
export function selectionAfterDuplicateRemoval({ dataset, currentSelection = [], removeIds = [] }) {
  const removeSet = new Set(removeIds);
  const removeKeys = new Set();
  (dataset?.items ?? []).forEach((item, index) => {
    if (removeSet.has(item.id)) {
      removeKeys.add(datasetItemSelectionKey(dataset, item, index));
    }
  });
  return {
    nextSelection: (currentSelection ?? []).filter((key) => !removeKeys.has(key)),
    removedCount: removeKeys.size,
  };
}

export function datasetHealth({ activeDataset, imageAssets, selectedAssetIds }) {
  const assetsById = new Map(imageAssets.map((asset) => [asset.id, asset]));
  const selectedAssets = selectedAssetIds.map((id) => assetsById.get(id)).filter(Boolean);
  const missingAssets = selectedAssetIds.filter((id) => !assetsById.has(id)).length;
  const disabledItems = selectedAssets.filter((asset) => asset.status?.rejected || asset.status?.trashed).length + missingAssets;
  const names = selectedAssets.map((asset) => imageAssetName(asset).toLowerCase());
  const duplicateFilenames = names.filter((name, index) => names.indexOf(name) !== index).length;
  const captionsByAssetId = new Map(
    (activeDataset?.items ?? []).map((item, index) => [datasetItemSelectionKey(activeDataset, item, index), captionText(item)]),
  );
  const missingCaptions = selectedAssetIds.filter((id) => !captionsByAssetId.get(id)).length;
  const valid = selectedAssetIds.length > 0 && disabledItems === 0;

  return {
    disabledItems,
    duplicateFilenames,
    itemCount: selectedAssetIds.length,
    missingCaptions,
    valid,
  };
}

// What gates Save on the dataset editor, in the app-wide vocabulary (epic 10644). The
// `health` counts (missing captions, duplicate filenames) are deliberately NOT issues:
// they're advisory in nature but the DatasetHealth grid already renders them as counts,
// so a chip would just repeat what the grid shows. This rule set carries only what
// blocks Save.
//
// A missing name is a `requirement` — the empty field speaks for itself. Empty datasets
// are valid because Parquet import needs a persisted destination before it can materialize
// images. `disabledItems` is the one real `error`: assets that went unavailable after
// they were selected.
export function datasetSaveValidation({ name, selectedAssetIds }, { health } = {}) {
  const issues = [];
  if (!name?.trim()) {
    issues.push(issue.requirement("datasetName", "Name the dataset"));
  }
  if (selectedAssetIds?.length && health?.disabledItems > 0) {
    const n = health.disabledItems;
    issues.push(
      issue.error(
        null,
        `Remove ${n} unavailable image${n === 1 ? "" : "s"} — ${n === 1 ? "it has" : "they have"} been rejected, trashed, or deleted`,
      ),
    );
  }
  return issues;
}

function datasetItemExtras(item) {
  if (!item) return {};
  const extra = { ...item };
  for (const key of [
    "id", "assetId", "path", "displayName", "caption", "controlImagePath", "width", "height",
    "contentHash", "tier0Scalars", "qualityAck", "addedAt",
    // The edit-pair reference list is rebuilt from the draft on every save (sc-24161); a stale
    // copy of the stored one must never ride along and override it.
    "references",
  ]) {
    delete extra[key];
  }
  return extra;
}

export function datasetPayload({
  activeDataset,
  assetsById,
  associatedCharacterId,
  captionDraftById = {},
  referenceDraftById = {},
  name,
  selectedAssetIds,
}) {
  const itemsByAssetId = new Map(
    (activeDataset?.items ?? []).map((item, index) => [datasetItemSelectionKey(activeDataset, item, index), item]),
  );
  return {
    name: name.trim(),
    modality: "image",
    // sc-2022: associate the dataset with a character when one is set (created
    // from a character's images, or images imported from the Character tab).
    ...(associatedCharacterId ? { characterId: associatedCharacterId } : {}),
    items: selectedAssetIds
      .map((selectionId) => {
        const asset = assetsById.get(selectionId);
        if (!asset) {
          return null;
        }
        const previous = itemsByAssetId.get(selectionId);
        const draft = captionDraftById[selectionId];
        let caption;
        if (draft && (String(draft.text ?? "").length || draft.source)) {
          caption = {
            text: draft.text ?? "",
            source: draft.source ?? "manual",
            triggerWords: previous?.caption?.triggerWords ?? [],
          };
        } else if (previous?.caption) {
          caption = {
            text: previous.caption.text ?? "",
            source: previous.caption.source ?? "manual",
            triggerWords: previous.caption.triggerWords ?? [],
          };
        }
        const source = asset.datasetOwned || asset.datasetOnly ? { path: asset.file?.path } : { assetId: asset.id };
        // Instruction-edit pairs (sc-24161): the ORDERED references, each sent with the same source
        // contract as an item (library assetId, else the project-relative path of a dataset-owned or
        // freshly uploaded image). Order is kept verbatim — the engine numbers the references.
        const references = (referenceDraftById[selectionId] ?? [])
          .map((referenceId) => assetsById.get(referenceId))
          .filter(Boolean)
          .map((reference) => ({
            ...(reference.datasetOwned || reference.datasetOnly
              ? { path: reference.file?.path }
              : { assetId: reference.id }),
            displayName: reference.displayName ?? imageAssetName(reference),
          }));
        return {
          ...datasetItemExtras(previous),
          ...(previous?.id ? { id: previous.id } : {}),
          ...(previous?.controlImagePath ? { controlImagePath: previous.controlImagePath } : {}),
          ...source,
          displayName: asset.displayName ?? imageAssetName(asset),
          caption,
          ...(references.length ? { references } : {}),
        };
      })
      .filter(Boolean),
  };
}

// ---------------------------------------------------------------------------------------------
// Instruction-edit pairs (sc-24161, epic 24107). An edit-pair dataset item is the item image (the
// edit TARGET) + its caption (the edit INSTRUCTION) + 1..N ORDERED reference images, stored on the
// server item as `references: [{ assetId?, path, displayName? }]` in the order given.
// ---------------------------------------------------------------------------------------------

// The selection key the studio tracks a stored reference under: its library asset id while that
// asset is still in the catalog, else a synthetic key backed by a dataset-owned asset (mirrors
// `datasetItemSelectionKey` / `datasetOwnedAssets` for items).
export function datasetReferenceKey(dataset, item, index, reference, catalogIds = new Set()) {
  if (reference?.assetId && catalogIds.has(reference.assetId)) {
    return reference.assetId;
  }
  return `dataset-ref:${dataset?.id ?? "draft"}:${item?.id ?? "item"}:${index}`;
}

// Ordered reference drafts keyed by item selection id, seeded from the saved dataset.
export function referenceDraftsFromDataset(dataset, catalogAssets = []) {
  const catalogIds = new Set(catalogAssets.map((asset) => asset.id));
  const map = {};
  (dataset?.items ?? []).forEach((item, itemIndex) => {
    const references = Array.isArray(item?.references) ? item.references : [];
    if (!references.length) return;
    map[datasetItemSelectionKey(dataset, item, itemIndex)] = references.map((reference, index) =>
      datasetReferenceKey(dataset, item, index, reference, catalogIds),
    );
  });
  return map;
}

// Synthetic dataset-owned assets for stored references that are not (or no longer) catalog assets,
// so the editor can render and re-send them (their project-relative path re-materializes on save).
export function datasetReferenceAssets(dataset, projectId, catalogAssets = []) {
  const catalogIds = new Set(catalogAssets.map((asset) => asset.id));
  const owned = [];
  (dataset?.items ?? []).forEach((item) => {
    (Array.isArray(item?.references) ? item.references : []).forEach((reference, index) => {
      const id = datasetReferenceKey(dataset, item, index, reference, catalogIds);
      if (id === reference?.assetId) return;
      const path = datasetItemProjectPath(dataset, reference);
      if (!path) return;
      owned.push({
        id,
        assetId: reference.assetId ?? null,
        datasetOwned: true,
        projectId,
        type: "image",
        displayName: reference.displayName ?? imageAssetName(reference),
        file: {
          path,
          mimeType: `image/${String(path).split(".").pop() || "png"}`,
          width: reference.width ?? null,
          height: reference.height ?? null,
        },
      });
    });
  });
  return owned;
}

// Whether two ordered reference drafts differ for any of the given selection ids.
export function referenceDraftsDiffer(current = {}, saved = {}, selectionIds = []) {
  return selectionIds.some((id) => {
    const left = current[id] ?? [];
    const right = saved[id] ?? [];
    return left.length !== right.length || left.some((value, index) => value !== right[index]);
  });
}

// How many ordered references one edit-pair item may carry for a training target, or 0 when the
// target cannot train edit pairs. The target declares its cap (`limits.maxReferenceImages`, the
// Rust-owned contract the API enforces) and the base MODEL declares its own
// (`limits.maxReferenceAssets`, the render-side cap the Image Editor reads); the UI takes the lower
// of the two, so it can never offer more references than either side accepts. Qwen Image 2.1: 10.
export function trainingTargetReferenceCap(target, models = []) {
  const declared = Number(target?.limits?.maxReferenceImages);
  if (!Number.isInteger(declared) || declared <= 0) {
    return 0;
  }
  const model = (models ?? []).find((entry) => entry?.id === target?.baseModel);
  return Math.min(declared, maxReferencesForModel(model, declared));
}

// The largest per-item reference count any available training target accepts — what the dataset
// editor (which is target-agnostic until a run picks one) lets an item carry. 0 hides the editor's
// edit-pair affordances entirely.
export function datasetReferenceCap(targets = [], models = []) {
  return (targets ?? []).reduce(
    (cap, target) => Math.max(cap, trainingTargetReferenceCap(target, models)),
    0,
  );
}

// Append picked references to an item's ordered list: de-duplicated, never the item's own image,
// and never past `cap`. Returns the next list plus how many picks did not fit.
export function appendReferences(current = [], picked = [], { cap = 0, itemId = "" } = {}) {
  const next = [...current];
  let dropped = 0;
  for (const id of picked ?? []) {
    if (!id || id === itemId || next.includes(id)) continue;
    if (next.length >= cap) {
      dropped += 1;
      continue;
    }
    next.push(id);
  }
  return { next, dropped };
}

// The edit-pair dataset-shape issues for a training target, mirroring the API's plan-time floor
// (`validate_dataset_shape_for_target`) so Start training is held with the same reasons the server
// would refuse with. `cap` is the target's effective reference cap (0 = not an edit target).
export function editPairDatasetIssues(dataset, target, cap) {
  const items = dataset?.items ?? [];
  if (!target || !items.length) return [];
  const label = target.ui?.label ?? target.name ?? target.id;
  const name = (item) => item?.displayName || item?.id || "an item";
  const count = (item) => (Array.isArray(item?.references) ? item.references.length : 0);
  if (!cap) {
    const editItem = items.find((item) => count(item) > 0);
    return editItem
      ? [
          issue.error(
            "target",
            `This dataset has edit pairs (reference images on “${name(editItem)}”), but ${label} trains on captioned images only. Pick an edit training target.`,
          ),
        ]
      : [];
  }
  const issues = [];
  const bare = items.find((item) => count(item) === 0);
  if (bare) {
    issues.push(
      issue.error(
        "dataset",
        `${label} trains on edit pairs: every item needs at least one reference image (“${name(bare)}” has none).`,
      ),
    );
  }
  const over = items.find((item) => count(item) > cap);
  if (over) {
    issues.push(
      issue.error(
        "dataset",
        `“${name(over)}” has ${count(over)} reference images; ${label} accepts at most ${cap} per edit.`,
      ),
    );
  }
  const blank = items.find((item) => count(item) > 0 && !captionText(item));
  if (blank) {
    issues.push(issue.error("dataset", `“${name(blank)}” needs an edit instruction (its caption).`));
  }
  return issues;
}
