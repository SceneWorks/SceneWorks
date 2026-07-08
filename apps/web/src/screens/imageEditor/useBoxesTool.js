import { useEffect, useRef, useState } from "react";
import {
  BOX_PALETTE,
  MIN_BOX_PX,
  rectFromPoints,
  clampRectToCanvas,
  makeBox,
} from "./boxGeometry.js";

// Box layout tool (sc-6090) extracted from ImageEditor.jsx (sc-9752, F-052 follow-up).
// Owns the colored-box overlay state + its Konva node/transformer refs + the id sequence,
// the draw/select/drag/transform pointer + node handlers, and the transformer-binding
// effect. Behavior-preserving.
//
// REF-MIRROR SEMANTICS (the crux of this extraction):
//   - `boxesRef` / `boxColorRef` mirror `boxes` / `boxColor` into refs, synced by two
//     effects that are moved here VERBATIM (same [boxes] / [boxColor] deps). The editor's
//     undo `captureSnapshot` reads these refs synchronously to snapshot the pre-op box
//     state without a stale-closure; the editor's `applyHistoryAux` writes them back on a
//     restore. Both refs are returned so the editor keeps reading/writing the exact same
//     ref objects it did inline.
//   - `boxIdRef` is the monotonic session-unique id source; it is snapshotted (boxIdSeq)
//     and restored so a box added after an undo can't recycle an id a redo brings back.
//     Returned so the editor's snapshot/restore reads/writes the same ref.
//   - `boxDrawingRef` / `boxStartRef` hold the live drag gesture, read inside the pointer
//     handlers (which fire outside React's render) and by the editor's `escapeGesture`.
//     `boxDrawingRef` is returned for escapeGesture + resetEditorOverlays.
//   - `boxNodeRefs` maps box id → Konva node for transformer binding; cleared by the
//     editor's resetEditorOverlays. Returned so that reset stays identical.
//   - `boxTransformerRef` is bound to the selected box's node by the effect here.
//
// `working` and `tool` are read live inside the pointer handlers + the transformer
// effect; the editor feeds them from render scope so behavior/deps are unchanged. The
// pure `checkpoint`, `stagePointToImage`, and `setTool` are supplied by the editor.
export function useBoxesTool({ working, tool, checkpoint, stagePointToImage, setTool }) {
  // Box layout tool (sc-6090): colored rectangles drawn over the working image in
  // image-pixel coords. They drive the color-keyed edit path (sc-6093) and the
  // Ideogram bbox path (sc-6095). Session-only overlay state — boxes are not baked
  // into the working bitmap here, so they don't mark the session dirty.
  const [boxes, setBoxes] = useState([]); // [{ id, rect, color, type, desc, text, colorPalette }]
  const [selectedBoxId, setSelectedBoxId] = useState(null);
  const [boxColor, setBoxColor] = useState(BOX_PALETTE[0].value);
  const [boxDraft, setBoxDraft] = useState(null); // live rect during a drag-draw
  const boxDrawingRef = useRef(false);
  const boxStartRef = useRef(null);
  const boxIdRef = useRef(0);
  const boxNodeRefs = useRef(new Map()); // id → Konva node, for transformer binding
  const boxTransformerRef = useRef(null);

  // Live mirrors of the snapshot-relevant box state (see the ref-mirror note above).
  const boxesRef = useRef(boxes);
  const boxColorRef = useRef(boxColor);
  useEffect(() => { boxesRef.current = boxes; }, [boxes]);
  useEffect(() => { boxColorRef.current = boxColor; }, [boxColor]);

  function selectBoxTool() {
    if (working) setTool("boxes");
  }

  const nextBoxId = () => `box_${(boxIdRef.current += 1)}`;

  // Konva node registry so the transformer can bind to the selected box; the ref
  // callback removes a node when its box unmounts (tool switch / delete).
  const registerBoxNode = (id, node) => {
    if (node) boxNodeRefs.current.set(id, node);
    else boxNodeRefs.current.delete(id);
  };

  function boxPointerDown(event) {
    if (tool !== "boxes" || !working) return;
    // Only a click on the canvas background starts a new box — clicks on an
    // existing box (select/drag) or a transformer handle (resize) are left alone.
    const stage = event.target.getStage();
    const name = event.target?.name?.() ?? "";
    const onBackground = event.target === stage || name === "editor-image" || name === "editor-bg";
    if (!onBackground) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    boxDrawingRef.current = true;
    boxStartRef.current = pt;
    setSelectedBoxId(null);
    setBoxDraft({ x: pt.x, y: pt.y, width: 0, height: 0 });
  }

  function boxPointerMove(event) {
    if (!boxDrawingRef.current) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    setBoxDraft(rectFromPoints(boxStartRef.current, pt));
  }

  function boxPointerUp() {
    if (!boxDrawingRef.current) return;
    boxDrawingRef.current = false;
    const draft = boxDraft;
    setBoxDraft(null);
    // Discard a click / sub-minimum smudge; otherwise commit a new colored box.
    if (!draft || draft.width < MIN_BOX_PX || draft.height < MIN_BOX_PX) return;
    const rect = clampRectToCanvas(draft, working.width, working.height);
    const id = nextBoxId();
    checkpoint();
    setBoxes((prev) => [...prev, makeBox(id, rect, boxColor)]);
    setSelectedBoxId(id);
  }

  const updateBoxRect = (id, rect) =>
    setBoxes((prev) => prev.map((box) => (box.id === id ? { ...box, rect } : box)));

  // Patch a box's metadata (sc-6091): type / desc / text / colorPalette.
  const updateBox = (id, patch) =>
    setBoxes((prev) => prev.map((box) => (box.id === id ? { ...box, ...patch } : box)));

  function handleBoxDragEnd(id, event) {
    const node = event.target;
    const rect = clampRectToCanvas(
      { x: node.x(), y: node.y(), width: node.width(), height: node.height() },
      working.width,
      working.height,
    );
    node.setAttrs(rect);
    checkpoint();
    updateBoxRect(id, rect);
  }

  function handleBoxTransformEnd(id, event) {
    const node = event.target;
    const rect = clampRectToCanvas(
      { x: node.x(), y: node.y(), width: node.width() * node.scaleX(), height: node.height() * node.scaleY() },
      working.width,
      working.height,
    );
    node.scaleX(1);
    node.scaleY(1);
    node.setAttrs(rect);
    checkpoint();
    updateBoxRect(id, rect);
  }

  // Selecting a palette color sets the color for new boxes and recolors the
  // selected box (the palette acts on the active box). Stored uppercase so the
  // box stays valid per `isValidHexColor` even from a lowercase <input type=color>.
  function chooseBoxColor(color) {
    const value = color.toUpperCase();
    // Recoloring the active box is an undoable step; setting the color for future
    // boxes (no selection) is not — it changes no committed state.
    if (selectedBoxId) checkpoint();
    setBoxColor(value);
    if (selectedBoxId) {
      setBoxes((prev) => prev.map((box) => (box.id === selectedBoxId ? { ...box, color: value } : box)));
    }
  }

  function deleteBox(id) {
    if (!id) return;
    checkpoint();
    setBoxes((prev) => prev.filter((box) => box.id !== id));
    boxNodeRefs.current.delete(id);
    setSelectedBoxId((cur) => (cur === id ? null : cur));
  }

  function clearBoxes() {
    if (boxes.length) checkpoint();
    setBoxes([]);
    boxNodeRefs.current.clear();
    setSelectedBoxId(null);
    setBoxDraft(null);
  }

  // Reset the per-bitmap box overlay state (called by the editor's resetEditorOverlays).
  // Mirrors the exact box-clearing lines from the inline resetEditorOverlays.
  function resetBoxState() {
    setBoxes([]);
    setSelectedBoxId(null);
    setBoxDraft(null);
    boxNodeRefs.current.clear();
    boxDrawingRef.current = false;
  }

  // Bind the transformer to the selected box whenever the box tool is active.
  useEffect(() => {
    const transformer = boxTransformerRef.current;
    if (tool !== "boxes" || !transformer) return;
    const node = selectedBoxId ? boxNodeRefs.current.get(selectedBoxId) : null;
    transformer.nodes(node ? [node] : []);
    transformer.getLayer()?.batchDraw();
  }, [tool, selectedBoxId, boxes]);

  return {
    // State
    boxes,
    selectedBoxId,
    boxColor,
    boxDraft,
    // Setters used in render / cross-cutting plumbing
    setBoxes,
    setSelectedBoxId,
    setBoxColor,
    setBoxDraft,
    // Refs the editor's snapshot/restore/escape/reset plumbing reads or writes directly
    boxesRef,
    boxColorRef,
    boxIdRef,
    boxDrawingRef,
    boxNodeRefs,
    boxTransformerRef,
    // Handlers
    selectBoxTool,
    registerBoxNode,
    boxPointerDown,
    boxPointerMove,
    boxPointerUp,
    updateBox,
    handleBoxDragEnd,
    handleBoxTransformEnd,
    chooseBoxColor,
    deleteBox,
    clearBoxes,
    resetBoxState,
  };
}
