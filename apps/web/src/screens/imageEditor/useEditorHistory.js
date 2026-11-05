import { emptyHistory, historyCheckpoint, historyUndo, historyRedo, canUndo, canRedo, runGuardedRestore } from "./history.js";
import { blobToImage } from "../../imageDimensions.js";
import { useCallback, useRef, useState } from "react";
import { createLayer, sameLayerStack, snapshotLayers } from "../../imageLayers.js";

// Extracted from ImageEditor along its existing concern boundary.
export function useEditorHistory({
  workingRef,
  layerIdRef,
  editsRef,
  dirtyRef,
  savedAssetIdRef,
  boxesRef,
  boxColorRef,
  boxIdRef,
  checkpointRef,
  setEdits,
  setDirty,
  setSavedAssetId,
  setBoxes,
  setBoxColor,
  setSelectedBoxId,
  resetEditorOverlays,
  needsFitRef,
  setWorking,
  setStatus,
  aiOpRef
}) {
  // Undo/redo (sc-6106): a bounded snapshot history over the working-image session.
  // The stacks live in a ref for synchronous reads inside the commit handlers; the
  // can-undo/redo flags are mirrored into state so the toolbar buttons re-render.
  const historyRef = useRef(emptyHistory());

  // Serializes undo/redo (sc-8852): a restore is async (it decodes changed layers
  // via blobToImage before re-installing the stack), so a rapid second Cmd-Z would
  // otherwise race the in-flight restore and corrupt the redo branch. Shared by
  // both undo() and redo() via runGuardedRestore — set while restoring, cleared in
  // a finally so a mid-restore error can't wedge history forever.
  const isRestoringRef = useRef(false);

  const [historyFlags, setHistoryFlags] = useState({ canUndo: false, canRedo: false });

  // ── Undo/redo plumbing (sc-6106, extended to the layer stack sc-6117) ──────
  // A snapshot is the layer stack (each layer as metadata + its shared blob, no
  // live image/URL) plus the overlay/provenance state that a re-install would
  // otherwise reset (boxes, edit chain, dirty flag). Blobs are shared by reference
  // across snapshots, so retained bitmaps stay bounded like the single-bitmap days.
  const captureSnapshot = useCallback(() => {
    const work = workingRef.current;
    return {
      layers: snapshotLayers(work?.layers),
      activeLayerId: work?.activeLayerId ?? null,
      width: work?.width ?? 0,
      height: work?.height ?? 0,
      source: work?.source ?? null,
      layerIdSeq: layerIdRef.current,
      edits: editsRef.current,
      dirty: dirtyRef.current,
      savedAssetId: savedAssetIdRef.current,
      boxes: boxesRef.current,
      boxColor: boxColorRef.current,
      boxIdSeq: boxIdRef.current,
    };
    // boxesRef / boxColorRef / boxIdRef are stable refs (from useBoxesTool); empty deps
    // preserve the pre-extraction behavior (a checkpoint reads them live, sc-9752).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const syncHistoryFlags = useCallback(() => {
    setHistoryFlags({ canUndo: canUndo(historyRef.current), canRedo: canRedo(historyRef.current) });
  }, []);

  // Record a step: push the pre-operation snapshot onto the undo stack. Call this
  // BEFORE the working state mutates (crop/color/AI result, layer op, or a box change).
  const checkpoint = useCallback(() => {
    if (!workingRef.current) return;
    historyRef.current = historyCheckpoint(historyRef.current, captureSnapshot());
    syncHistoryFlags();
  }, [captureSnapshot, syncHistoryFlags, workingRef]);

  // Keep the tool hooks' bridge pointed at the latest checkpoint (sc-9752).
  checkpointRef.current = checkpoint;

  // Start a fresh history for a newly opened session (clears both stacks).
  const resetHistory = useCallback(() => {
    historyRef.current = emptyHistory();
    syncHistoryFlags();
  }, [syncHistoryFlags]);

  // Re-apply a snapshot's overlay/provenance state, keeping the live mirrors in
  // sync immediately so an undo→undo chain reads the right "present" each step.
  const applyHistoryAux = useCallback((snap) => {
    setEdits(snap.edits);
    editsRef.current = snap.edits;
    setDirty(snap.dirty);
    dirtyRef.current = snap.dirty;
    setSavedAssetId(snap.savedAssetId);
    savedAssetIdRef.current = snap.savedAssetId;
    setBoxes(snap.boxes);
    boxesRef.current = snap.boxes;
    setBoxColor(snap.boxColor);
    boxColorRef.current = snap.boxColor;
    setSelectedBoxId(null);
    boxIdRef.current = snap.boxIdSeq;
    // Restore the layer-id counter so a layer added after this undo can't recycle
    // an id that a redo would bring back (mirrors boxIdSeq).
    if (typeof snap.layerIdSeq === "number") layerIdRef.current = snap.layerIdSeq;
    // The box setters + refs come from useBoxesTool but are stable (useState setters +
    // useRefs); empty deps preserve the pre-extraction behavior (sc-9752).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const restoreSnapshot = useCallback(
    async (snap) => {
      if (!snap) return;
      try {
        const live = workingRef.current;
        // Overlay-only steps (box edits) keep the stack pixel- and metadata-identical
        // → skip the rebuild, the decode, and the view refit; only a bitmap/structure
        // change (crop/color/AI/layer op) re-installs the stack.
        const stackChanged =
          !live ||
          !sameLayerStack(live.layers, snap.layers) ||
          live.activeLayerId !== snap.activeLayerId ||
          live.width !== snap.width ||
          live.height !== snap.height;
        if (stackChanged) {
          // Rebuild the live stack from the snapshot: reuse a live layer's decoded
          // image when its blob is unchanged (decode ONLY changed/new layers), and
          // revoke the object URLs of live layers the restore drops.
          const liveById = new Map((live?.layers ?? []).map((layer) => [layer.id, layer]));
          const reused = new Set();
          const layers = [];
          for (const sl of snap.layers) {
            const prev = liveById.get(sl.id);
            if (prev && prev.blob === sl.blob && prev.image) {
              reused.add(sl.id);
              layers.push({
                ...prev,
                name: sl.name,
                visible: sl.visible,
                opacity: sl.opacity,
                blendMode: sl.blendMode,
                transform: { ...sl.transform },
              });
            } else {
              const { image, objectUrl } = await blobToImage(sl.blob);
              layers.push(createLayer({ ...sl, image, objectUrl }));
            }
          }
          for (const layer of live?.layers ?? []) {
            if (!reused.has(layer.id) && layer.objectUrl) URL.revokeObjectURL(layer.objectUrl);
          }
          // A true PIXEL change = dims changed, a layer added/removed, or a layer's
          // blob differs. Metadata-only undos (opacity / visibility / blend / transform
          // / reorder — same id→blob set, same dims) keep the current tool, mask, boxes
          // and view; only a pixel change resets the per-bitmap overlays + refits.
          const dimsChanged = !live || live.width !== snap.width || live.height !== snap.height;
          const liveBlobs = new Map((live?.layers ?? []).map((layer) => [layer.id, layer.blob]));
          const bitmapChanged =
            dimsChanged ||
            liveBlobs.size !== snap.layers.length ||
            snap.layers.some((sl) => liveBlobs.get(sl.id) !== sl.blob);
          if (bitmapChanged) resetEditorOverlays();
          if (dimsChanged) needsFitRef.current = true;
          const nextWorking = {
            width: snap.width,
            height: snap.height,
            source: snap.source,
            layers,
            activeLayerId: snap.activeLayerId,
          };
          workingRef.current = nextWorking;
          setWorking(nextWorking);
        }
        applyHistoryAux(snap);
      } catch (err) {
        setStatus({ loading: false, error: err.message || "Could not restore that step." });
      }
    },
    [workingRef, applyHistoryAux, resetEditorOverlays, needsFitRef, setWorking, setStatus],
  );

  const undo = useCallback(async () => {
    if (aiOpRef.current || !workingRef.current) return;
    await runGuardedRestore({
      guardRef: isRestoringRef,
      step: () => historyUndo(historyRef.current, captureSnapshot()),
      commitHistory: (next) => {
        historyRef.current = next;
        syncHistoryFlags();
      },
      restore: restoreSnapshot,
    });
  }, [aiOpRef, captureSnapshot, restoreSnapshot, syncHistoryFlags, workingRef]);

  const redo = useCallback(async () => {
    if (aiOpRef.current || !workingRef.current) return;
    await runGuardedRestore({
      guardRef: isRestoringRef,
      step: () => historyRedo(historyRef.current, captureSnapshot()),
      commitHistory: (next) => {
        historyRef.current = next;
        syncHistoryFlags();
      },
      restore: restoreSnapshot,
    });
  }, [aiOpRef, captureSnapshot, restoreSnapshot, syncHistoryFlags, workingRef]);
  return { historyFlags, checkpoint, resetHistory, undo, redo };
}
