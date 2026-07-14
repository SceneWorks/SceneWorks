import React, { useState } from "react";
import { apiFetch } from "./api.js";
import { batchEligibleAssets, batchItemStatus, buildBatchJob, summarizeBatchProgress } from "./batchOps.js";
import { upscaledFromAssetId } from "./assetVariants.js";
import { assetSupportsCharacterLink } from "./components/assetPanels.jsx";
import { assetUrl } from "./components/assetMedia.jsx";
import { BatchOperationsPanel } from "./components/BatchOperationsPanel.jsx";
import { Modal } from "./components/Modal.jsx";
import { useAppContextOptional } from "./context/AppContext.js";
import { detailCapableModels, editCapableModels, UPSCALE_ENGINES } from "./imageJobs.js";
import { DEFAULT_MAC_CAPABILITIES, macUpscaleEngineBlocked } from "./macGating.js";

// Sentinel "Move" target (sc-8341): selecting it promotes the assets into the Main Asset
// Library (a true move) instead of linking them to a character. Namespaced so it can't
// collide with a real character id.
export const LIBRARY_MOVE_TARGET = "__sceneworks_library__";

// Shared multi-asset batch selection (sc-6112) — selection state, the upscale/detail/edit
// fan-out, and the bulk Discard / Move-to-character actions. Lifted out of LibraryScreen so
// the Assets page and the Character Assets page drive the identical toolbar from one source.
// The hook owns the selection; callers wire `selectedAssetIds`/`toggleSelect` into their grid
// and render <AssetSelectionBar batch={…}/> + <AssetBatchModal batch={…}/>.
export function useAssetBatch() {
  // Read the app context optionally: the toolbar should stay inert (not crash) when a host
  // component is rendered in isolation without any provider (e.g. unit tests). The app
  // renders the split AppStaticContext/AppLiveContext providers (not the legacy combined
  // <AppContext.Provider>), so we read the merged view across both — useContext(AppContext)
  // alone would resolve to null in the real app and blank the toolbar. `jobs` is a live
  // field, so this still (correctly) re-renders the toolbar on job ticks.
  const {
    activeProject,
    assets = [],
    jobs = [],
    imageModels = [],
    characters = [],
    deleteAsset,
    moveAssetToCharacter,
    moveAssetToLibrary,
    token = "",
    requestedGpu = "auto",
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
  } = useAppContextOptional() ?? {};

  const [selectedAssetIds, setSelectedAssetIds] = useState(() => new Set());
  const [batchOpen, setBatchOpen] = useState(false);
  // While/after a batch runs: { op, items: [{ asset, jobId }], submitting }.
  const [batch, setBatch] = useState(null);
  // Bulk Discard / Move-to-character on the current selection. `bulkAction` gates the
  // buttons while a fan-out is in flight; `moveOpen` reveals the inline character picker.
  const [bulkAction, setBulkAction] = useState(null);
  // When a discard selection contains folded upscales, hold the pending choice here:
  // { targets: Asset[] (the selection snapshot), sources: Asset[] (their source originals) }.
  // Non-null drives the DiscardUpscaledDialog; the user picks both / upscaled-only / cancel.
  const [discardPrompt, setDiscardPrompt] = useState(null);
  const [moveOpen, setMoveOpen] = useState(false);
  const [moveCharacterId, setMoveCharacterId] = useState("");

  // The current multi-selection, narrowed to the raster images a batch op can run on,
  // and the upscale engines this platform actually supports.
  const selectedAssetList = assets.filter((asset) => selectedAssetIds.has(asset.id));
  const eligibleSelected = batchEligibleAssets(selectedAssetList);
  const availableUpscaleEngines = UPSCALE_ENGINES.filter((engine) => !macUpscaleEngineBlocked(macCapabilities, engine.key));
  const editModels = editCapableModels(imageModels);
  const detailModels = detailCapableModels(imageModels);
  // Move targets a character's assets (true move, sc-10200) — NOT the character's
  // curated reference set — so only move-capable media counts.
  const availableCharacters = characters.filter((character) => !character?.archived);
  const movableSelected = selectedAssetList.filter(assetSupportsCharacterLink);

  // Per-item + aggregate progress for an in-flight/just-finished batch, read off the jobs feed.
  const batchItems = batch
    ? batch.items.map((item) => ({ asset: item.asset, status: batchItemStatus(item.jobId, jobs) }))
    : null;
  const batchProgress = batch ? summarizeBatchProgress(batch.items, jobs) : null;

  const toggleSelect = (id) =>
    setSelectedAssetIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  // Add every id in `ids` to the selection (union — never drops an existing pick), so a
  // "Select all" over the current result set stacks onto anything already chosen.
  const selectAll = (ids) =>
    setSelectedAssetIds((prev) => {
      const next = new Set(prev);
      for (const id of ids) next.add(id);
      return next;
    });
  const clearSelection = () => {
    setSelectedAssetIds(new Set());
    setMoveOpen(false);
  };

  // Decode an asset's native pixel size (needed for an edit job — the worker fits the
  // source to width×height). Resolves null on a load failure so that item fails alone.
  function loadImageDims(asset) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = () => resolve(null);
      img.src = assetUrl(asset);
    });
  }

  // Fan out one job per selected image (NOT one mega-job): each posts independently so
  // the worker processes them serially with its between-item cache release (sc-5567).
  async function runBatch(op, params) {
    if (!activeProject || !eligibleSelected.length) return;
    const targets = eligibleSelected;
    setBatch({ op, submitting: true, items: targets.map((asset) => ({ asset, jobId: null })) });
    const items = [];
    for (const asset of targets) {
      try {
        let dims = null;
        if (op === "edit") {
          dims = await loadImageDims(asset);
          if (!dims) {
            items.push({ asset, jobId: null });
            continue;
          }
        }
        const { endpoint, body } = buildBatchJob({ op, asset, params, project: activeProject, requestedGpu, dims });
        const job = await apiFetch(endpoint, token, { method: "POST", body: JSON.stringify(body) });
        items.push({ asset, jobId: job?.id ?? null });
      } catch {
        items.push({ asset, jobId: null });
      }
    }
    setBatch({ op, submitting: false, items });
  }

  function closeBatch() {
    setBatchOpen(false);
    // Closing after a run clears the spent selection + progress; cancelling the form keeps it.
    if (batch) {
      setBatch(null);
      clearSelection();
    }
  }

  // Send the current selection to the Trash (reversible — the backend just flags `trashed`).
  // Selected tiles are the folded upscale representatives (foldUpscaledAssetVariants keeps the
  // upscaled asset as the tile and hides its source original), so trashing the selection alone
  // leaves each source behind — the user then has to re-select everything. When any selected
  // tile is an upscale whose source original is still present, confirm first whether to also
  // discard those originals; a selection with no such tiles trashes immediately as before (sc-10340).
  async function discardSelected() {
    if (!selectedAssetList.length || bulkAction) return;
    const byId = new Map(assets.map((asset) => [asset.id, asset]));
    const sources = [];
    const seenSourceIds = new Set();
    for (const asset of selectedAssetList) {
      const sourceId = upscaledFromAssetId(asset);
      const source = sourceId ? byId.get(sourceId) : null;
      // deleteAsset only flags `trashed` (the asset stays in `assets`), so a source that was
      // already discarded on its own still resolves here — skip those (and missing ones):
      // there's nothing new to trash, and offering it would fire a redundant delete.
      if (source && !source.status?.trashed && !seenSourceIds.has(sourceId)) {
        seenSourceIds.add(sourceId);
        sources.push(source);
      }
    }
    if (sources.length) {
      // Snapshot the selection + resolved originals; DiscardUpscaledDialog drives the choice.
      setDiscardPrompt({ targets: selectedAssetList, sources });
      return;
    }
    await performDiscard(selectedAssetList, []);
  }

  // Trash `targets` plus any `sources`, de-duped by id (a source that is also directly
  // selected must not be deleted twice), then clear the selection.
  async function performDiscard(targets, sources) {
    if (bulkAction) return;
    setBulkAction("discard");
    try {
      const toDelete = new Map();
      for (const asset of targets) toDelete.set(asset.id, asset);
      for (const asset of sources) toDelete.set(asset.id, asset);
      for (const asset of toDelete.values()) {
        await deleteAsset?.(asset);
      }
      clearSelection();
    } finally {
      setBulkAction(null);
    }
  }

  // Resolve an open discard prompt: `includeSources` trashes the source originals too;
  // otherwise only the selected (upscaled) tiles are trashed.
  function resolveDiscardPrompt(includeSources) {
    const pending = discardPrompt;
    setDiscardPrompt(null);
    if (pending) {
      performDiscard(pending.targets, includeSources ? pending.sources : []);
    }
  }

  function cancelDiscard() {
    setDiscardPrompt(null);
  }

  // Fan out the move across every movable selection. Both targets are TRUE moves:
  // the Main Asset Library (sc-8341) or a character's assets (sc-10200 — the asset
  // leaves the Library and every other character; it is NOT added to the target's
  // curated "Approved set").
  async function moveSelectedToCharacter() {
    if (!moveCharacterId || !movableSelected.length || bulkAction) return;
    const toLibrary = moveCharacterId === LIBRARY_MOVE_TARGET;
    if (toLibrary ? !moveAssetToLibrary : !moveAssetToCharacter) return;
    setBulkAction("move");
    try {
      for (const asset of movableSelected) {
        try {
          if (toLibrary) {
            await moveAssetToLibrary(asset);
          } else {
            await moveAssetToCharacter(asset, moveCharacterId);
          }
        } catch {
          // One asset failing (e.g. already moved) shouldn't abort the rest.
        }
      }
      clearSelection();
    } finally {
      setBulkAction(null);
    }
  }

  return {
    selectedAssetIds,
    toggleSelect,
    selectAll,
    clearSelection,
    selectedAssetList,
    eligibleSelected,
    movableSelected,
    availableCharacters,
    batchOpen,
    setBatchOpen,
    batch,
    batchItems,
    batchProgress,
    runBatch,
    closeBatch,
    bulkAction,
    discardSelected,
    discardPrompt,
    resolveDiscardPrompt,
    cancelDiscard,
    moveSelectedToCharacter,
    moveOpen,
    setMoveOpen,
    moveCharacterId,
    setMoveCharacterId,
    editModels,
    detailModels,
    availableUpscaleEngines,
  };
}

// The selection toolbar: appears once anything is selected. `showDiscard` lets a host hide
// the trash action where it doesn't apply (e.g. a Trashcan view, where items are already
// discarded). `allowLibraryTarget` adds "Assets Library" as a Move destination (sc-8341) —
// used on the Character Assets page to promote media back into the Main Library.
export function AssetSelectionBar({ batch, showDiscard = true, allowLibraryTarget = false }) {
  const {
    selectedAssetIds,
    eligibleSelected,
    movableSelected,
    availableCharacters,
    setBatchOpen,
    bulkAction,
    discardSelected,
    discardPrompt,
    resolveDiscardPrompt,
    cancelDiscard,
    moveOpen,
    setMoveOpen,
    moveCharacterId,
    setMoveCharacterId,
    moveSelectedToCharacter,
    clearSelection,
  } = batch;

  // Rendered independently of the toolbar's own visibility gate: the confirmation snapshots
  // its targets, so it must survive even if the selection state momentarily empties.
  const discardDialog = discardPrompt ? (
    <DiscardUpscaledDialog
      sourceCount={discardPrompt.sources.length}
      busy={bulkAction === "discard"}
      onDiscardBoth={() => resolveDiscardPrompt(true)}
      onDiscardUpscaledOnly={() => resolveDiscardPrompt(false)}
      onCancel={cancelDiscard}
    />
  ) : null;

  if (selectedAssetIds.size === 0) return discardDialog;

  // Move destinations: the Main Library (optional) followed by every non-archived character.
  const moveTargets = [
    ...(allowLibraryTarget ? [{ id: LIBRARY_MOVE_TARGET, name: "Assets Library" }] : []),
    ...availableCharacters,
  ];
  const toLibrary = moveCharacterId === LIBRARY_MOVE_TARGET;

  return (
    <div className="batch-selection-bar">
      <span>
        {selectedAssetIds.size} selected
        {eligibleSelected.length !== selectedAssetIds.size
          ? ` · ${eligibleSelected.length} image${eligibleSelected.length === 1 ? "" : "s"}`
          : ""}
      </span>
      <button className="primary" disabled={!eligibleSelected.length} onClick={() => setBatchOpen(true)} type="button">
        Batch…
      </button>
      {showDiscard ? (
        <button className="danger-action" disabled={Boolean(bulkAction)} onClick={discardSelected} type="button">
          {bulkAction === "discard" ? "Discarding…" : "Discard"}
        </button>
      ) : null}
      {moveTargets.length ? (
        <button
          disabled={!movableSelected.length || Boolean(bulkAction)}
          onClick={() =>
            setMoveOpen((open) => {
              const next = !open;
              if (next && !moveCharacterId) {
                setMoveCharacterId(moveTargets[0]?.id ?? "");
              }
              return next;
            })
          }
          title={movableSelected.length ? undefined : "No movable media selected"}
          type="button"
        >
          Move
        </button>
      ) : null}
      <button onClick={clearSelection} type="button">
        Clear
      </button>
      {moveOpen && moveTargets.length ? (
        <div className="batch-move-picker">
          <select
            aria-label="Move target"
            onChange={(event) => setMoveCharacterId(event.target.value)}
            value={moveCharacterId}
          >
            {moveTargets.map((target) => (
              <option key={target.id} value={target.id}>
                {target.name}
              </option>
            ))}
          </select>
          <button
            className="primary"
            disabled={!moveCharacterId || !movableSelected.length || Boolean(bulkAction)}
            onClick={moveSelectedToCharacter}
            type="button"
          >
            {bulkAction === "move"
              ? "Moving…"
              : `Move ${movableSelected.length} to ${toLibrary ? "library" : "assets"}`}
          </button>
          <button onClick={() => setMoveOpen(false)} type="button">
            Cancel
          </button>
        </div>
      ) : null}
      {discardDialog}
    </div>
  );
}

// Confirmation shown when a bulk Discard selection contains folded upscales: an upscaled
// tile hides its source original, so this asks whether to trash those originals too. Escape
// or a backdrop click (via the shared Modal) cancels and discards nothing. The three explicit
// outcomes — both / upscaled-only / cancel — avoid the OK/Cancel ambiguity of window.confirm
// (here "cancel" still leaves the upscaled undeleted, which OK/Cancel could not convey).
function DiscardUpscaledDialog({ sourceCount, busy, onDiscardBoth, onDiscardUpscaledOnly, onCancel }) {
  const plural = sourceCount === 1 ? "" : "s";
  return (
    <Modal className="discard-confirm-modal" labelledBy="discard-confirm-title" onClose={onCancel}>
      <h2 className="discard-confirm-title" id="discard-confirm-title">
        Discard source image{plural} too?
      </h2>
      <p className="discard-confirm-body">
        Your selection includes upscaled image{plural} with{" "}
        {sourceCount === 1 ? "a source original" : `${sourceCount} source originals`} still in your library.
        Discard the source original{plural} as well, or only the upscaled image{plural}?
      </p>
      <div className="discard-confirm-actions">
        <button disabled={busy} onClick={onCancel} type="button">
          Cancel
        </button>
        <button disabled={busy} onClick={onDiscardUpscaledOnly} type="button">
          Only the upscaled
        </button>
        <button className="danger-action" disabled={busy} onClick={onDiscardBoth} type="button">
          Discard both
        </button>
      </div>
    </Modal>
  );
}

// The batch-operations modal (upscale / detail / edit). Renders only while open.
export function AssetBatchModal({ batch }) {
  if (!batch.batchOpen) return null;
  return (
    <BatchOperationsPanel
      assets={batch.eligibleSelected}
      editModels={batch.editModels}
      detailModels={batch.detailModels}
      upscaleEngines={batch.availableUpscaleEngines}
      busy={Boolean(batch.batch?.submitting)}
      items={batch.batchItems}
      progress={batch.batchProgress}
      onRun={batch.runBatch}
      onClose={batch.closeBatch}
    />
  );
}
