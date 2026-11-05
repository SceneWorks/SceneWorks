import { blobToImage } from "../../imageDimensions.js";
import { useRef } from "react";
import { activeLayerOf, addLayer, createLayer, duplicateLayer, identityTransform, layerById, moveLayer, removeLayer, setActiveLayer, setLayerProps } from "../../imageLayers.js";

// Extracted from ImageEditor along its existing concern boundary.
export function useLayerOps({
  setWorking,
  workingRef,
  checkpoint,
  setDirty,
  nextLayerId,
  setStatus,
  working,
  setTool,
  imageNodeRef
}) {
  // One undo checkpoint per opacity DRAG: the first change of
  // a gesture snapshots; subsequent ticks coalesce until pointer-up resets it.
  const layerOpacityGestureRef = useRef(false);

  // One undo checkpoint per Transform slider DRAG (mirrors the opacity gesture).
  const transformGestureRef = useRef(false);

  // ── Layer stack ops (sc-6118) ─────────────────────────────────────────────
  // Wire the layers panel to the pure layer-stack ops (../imageLayers.js). Each
  // mutating op checkpoints first (sc-6106 → undoable) and marks the session dirty.
  // Structural ops manage object URLs: delete revokes the evicted layer's URL;
  // add/duplicate decode a fresh blob into the new layer's own image + URL.
  function selectLayer(id) {
    setWorking((prev) => (prev ? setActiveLayer(prev, id) : prev));
  }

  function toggleLayerVisible(id) {
    if (!workingRef.current) return;
    checkpoint();
    setWorking((prev) => {
      const layer = layerById(prev, id);
      return layer ? setLayerProps(prev, id, { visible: !layer.visible }) : prev;
    });
    setDirty(true);
  }

  // One undo step per opacity DRAG: the panel flags the first change of a gesture
  // (`isGestureStart`) → checkpoint once, then the rest of the drag just updates.
  function changeLayerOpacity(id, opacity, isGestureStart) {
    if (!workingRef.current) return;
    if (isGestureStart) checkpoint();
    setWorking((prev) => setLayerProps(prev, id, { opacity }));
    setDirty(true);
  }

  function renameLayer(id, name) {
    const work = workingRef.current;
    const layer = work && layerById(work, id);
    if (!layer || layer.name === name) return;
    checkpoint();
    setWorking((prev) => setLayerProps(prev, id, { name }));
    setDirty(true);
  }

  function reorderLayer(id, toIndex) {
    if (!workingRef.current) return;
    checkpoint();
    setWorking((prev) => moveLayer(prev, id, toIndex));
    setDirty(true);
  }

  function deleteLayer(id) {
    const work = workingRef.current;
    if (!work || work.layers.length <= 1) return;
    checkpoint();
    const { working: next, removed } = removeLayer(work, id);
    if (!removed) return;
    setWorking(next);
    if (removed.objectUrl) URL.revokeObjectURL(removed.objectUrl);
    setDirty(true);
  }

  async function duplicateLayerById(id) {
    const work = workingRef.current;
    const src = work && layerById(work, id);
    if (!src) return;
    const { image, objectUrl } = await blobToImage(src.blob);
    checkpoint();
    setWorking((prev) => duplicateLayer(prev, id, { id: nextLayerId(), image, objectUrl }));
    setDirty(true);
  }

  async function addBlankLayer() {
    const work = workingRef.current;
    if (!work) return;
    // A new transparent layer at the document size — a fresh surface above the
    // active layer. The tools begin targeting it with the sc-6119 per-layer matrix.
    const canvas = document.createElement("canvas");
    canvas.width = work.width;
    canvas.height = work.height;
    let blob;
    try {
      blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    } finally {
      canvas.width = 0;
      canvas.height = 0;
    }
    if (!blob) {
      setStatus({ loading: false, error: "Could not create the layer." });
      return;
    }
    const { image, objectUrl } = await blobToImage(blob);
    checkpoint();
    setWorking((prev) =>
      addLayer(
        prev,
        createLayer({ id: nextLayerId(), name: `Layer ${prev.layers.length + 1}`, image, objectUrl, blob }),
      ),
    );
    setDirty(true);
  }

  // Per-layer blend mode (sc-6120): metadata only — the layer's <KonvaImage> node +
  // the flatten compositor both apply it via globalCompositeOperation.
  function setLayerBlend(id, blendMode) {
    if (!workingRef.current) return;
    checkpoint();
    setWorking((prev) => setLayerProps(prev, id, { blendMode }));
    setDirty(true);
  }

  // ── Per-layer transform (sc-6120) ─────────────────────────────────────────
  // The Transform tool binds a Konva Transformer to the ACTIVE layer's node. The
  // node renders from `layer.transform` (x/y/scale/rotation); on drag/transform end
  // we read the node back into the layer's transform metadata. The bitmap is never
  // resampled — the transform is baked only at flatten time (compositeLayersToCanvas
  // already honors it, matching the live node 1:1).
  function startTransform() {
    if (working) setTool("transform");
  }

  function commitActiveLayerTransform() {
    const node = imageNodeRef.current;
    const layer = activeLayerOf(workingRef.current);
    if (!node || !layer) return;
    const transform = {
      x: node.x(),
      y: node.y(),
      scaleX: node.scaleX(),
      scaleY: node.scaleY(),
      rotation: node.rotation(),
    };
    checkpoint();
    setWorking((prev) => setLayerProps(prev, layer.id, { transform }));
    setDirty(true);
  }

  function resetActiveLayerTransform() {
    const layer = activeLayerOf(workingRef.current);
    if (!layer) return;
    checkpoint();
    setWorking((prev) => setLayerProps(prev, layer.id, { transform: identityTransform() }));
    setDirty(true);
  }

  // Numeric Transform controls (sc-10255): merge a patch into the active layer's
  // transform. Bound two-way to the same {x,y,scaleX,scaleY,rotation} the canvas
  // handles drive, so typing/sliding moves the layer and dragging updates the fields.
  // `gestureStart` gates the undo checkpoint so a slider drag is one step, not many.
  function setActiveTransform(patch, { gestureStart = true } = {}) {
    const layer = activeLayerOf(workingRef.current);
    if (!layer) return;
    if (gestureStart) checkpoint();
    setWorking((prev) => setLayerProps(prev, layer.id, { transform: { ...layer.transform, ...patch } }));
    setDirty(true);
  }

  const onTransformSlider = (patch) => {
    const start = !transformGestureRef.current;
    transformGestureRef.current = true;
    setActiveTransform(patch, { gestureStart: start });
  };

  const endTransformGesture = () => {
    transformGestureRef.current = false;
  };

  function flipActiveLayer(axis) {
    const layer = activeLayerOf(workingRef.current);
    if (!layer) return;
    const t = layer.transform;
    // Flip in place: negate the axis scale AND shift the origin by the scaled extent so
    // the layer keeps its local bounding box instead of mirroring off its top-left pivot.
    const patch =
      axis === "h"
        ? { scaleX: -t.scaleX, x: t.x + (layer.image?.naturalWidth ?? 0) * t.scaleX }
        : { scaleY: -t.scaleY, y: t.y + (layer.image?.naturalHeight ?? 0) * t.scaleY };
    setActiveTransform(patch);
  }
  return { layerOpacityGestureRef, selectLayer, toggleLayerVisible, changeLayerOpacity, renameLayer, reorderLayer, deleteLayer, duplicateLayerById, addBlankLayer, setLayerBlend, startTransform, commitActiveLayerTransform, resetActiveLayerTransform, setActiveTransform, onTransformSlider, endTransformGesture, flipActiveLayer };
}
