import { useEffect, useRef, useState } from "react";
import {
  maskAlphaFromRgba,
  writeMaskAlphaToRgba,
  invertAlpha,
  dilateAlpha,
  erodeAlpha,
  blurAlpha,
} from "../../maskRefine.js";
import { assetUrl } from "../../components/assetMedia.jsx";
import { MIN_BOX_PX, rectFromPoints, clampRectToCanvas } from "./boxGeometry.js";
import {
  buildSegmentJobBody,
  rectToSegmentBox,
  tintMaskRgbaInPlace,
  maskHasContent,
} from "./maskShared.js";

// Inpaint-mask brush + smart-select tool (sc-2436 / sc-3751 / sc-6110) extracted from
// ImageEditor.jsx (sc-9752, F-052 follow-up). Owns the brush-stroke + smart-select-base
// mask state, the brush/select pointer handlers, the mask rasterize / refine ops, the
// SAM3 smart-select run, and the "leave paint mode when the tool closes" effect.
// Behavior-preserving.
//
// REF-MIRROR SEMANTICS (the crux of this extraction):
//   - `maskPaintingRef` is set true on brush pointer-down and false on pointer-up; the
//     pointer-move handler (fired outside React render) reads it live to decide whether a
//     stroke is in progress. Moved here VERBATIM, still read only inside the brush handlers.
//   - `selectDrawingRef` / `selectStartRef` hold the live smart-select box drag: set on
//     select pointer-down, read on move (to span the rect from the start point) and up (to
//     commit), cleared on up / escape / reset. `selectDrawingRef` is returned so the editor's
//     `escapeGesture` + `resetEditorOverlays` can cancel an in-flight select gesture exactly
//     as before. These are NOT mirrored React state — they are gesture latches read live
//     inside the DOM pointer handlers, so keeping them as the same refs preserves behavior.
//   Unlike the boxes tool, the mask has no snapshot-mirror ref: mask state is transient (a
//   new bitmap resets it) and never enters the undo snapshot, so there is no boxesRef-style
//   mirror to keep — only the two gesture latches above.
//
// `working`, `tool`, `canMask`, `maskMode`, and `aiOp` are read live inside the handlers /
// effect; the editor feeds them from render scope so behavior + effect deps are unchanged.
export function useMaskTool({
  working,
  tool,
  canMask,
  aiOp,
  activeProject,
  requestedGpu,
  runAiOp,
  stagePointToImage,
  blobToImage,
  setTool,
}) {
  // Inpaint mask (sc-2436): freehand brush strokes in image-pixel coords, rasterized
  // to a mask asset on Run for inpaint-capable models. `maskMode` is the paint sub-mode
  // of the AI Edit tool (Stage panning is suspended while it's on).
  const [maskLines, setMaskLines] = useState([]); // [{ points:[x,y,…], size, erase }]
  const [maskMode, setMaskMode] = useState(false);
  const [maskBrush, setMaskBrush] = useState(64);
  const [maskErase, setMaskErase] = useState(false);
  const maskPaintingRef = useRef(false);
  // Mask refinement (sc-6110): feather / grow / shrink radius in px for the post-process ops.
  const [maskRefineRadius, setMaskRefineRadius] = useState(6);

  // Smart-select (sc-3751): a box-drag sub-mode of the mask tool that runs the SAM3 `image_segment`
  // job (sc-6105) and loads the returned binary mask as an editable base UNDER the brush strokes.
  // `maskBaseImage` is the white-on-black mask bitmap (rasterized into the mask PNG alongside the
  // strokes); `maskOverlay` is its pink-on-transparent preview (rendered in the mask layer, so the
  // eraser's destination-out clears it too). `maskSubTool` toggles brush vs. box-select.
  const [maskBaseImage, setMaskBaseImage] = useState(null); // HTMLImageElement | null
  const [maskOverlay, setMaskOverlay] = useState(null); // HTMLCanvasElement | null
  const [maskSubTool, setMaskSubTool] = useState("brush"); // "brush" | "select"
  const [selectDraft, setSelectDraft] = useState(null); // live selection rect during a drag
  const selectDrawingRef = useRef(false);
  const selectStartRef = useRef(null);

  // Leave paint mode (restoring Stage panning) when the edit tool is closed or the
  // model can't inpaint — otherwise the canvas would stay in a paint state with no UI.
  useEffect(() => {
    if (maskMode && (tool !== "edit" || !canMask)) setMaskMode(false);
  }, [tool, canMask, maskMode]);

  // Reset the per-bitmap mask state (called by the editor's resetEditorOverlays). Mirrors
  // the exact mask-clearing lines from the inline resetEditorOverlays.
  function resetMaskState() {
    setMaskLines([]);
    setMaskMode(false);
    setMaskBaseImage(null);
    setMaskOverlay(null);
    setMaskSubTool("brush");
    setSelectDraft(null);
    selectDrawingRef.current = false;
  }

  // ── Inpaint mask brush (sc-2436) ──────────────────────────────────────────
  function maskPointerDown(event) {
    if (!maskMode || maskSubTool !== "brush" || !working) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    maskPaintingRef.current = true;
    setMaskLines((prev) => [...prev, { points: [pt.x, pt.y], size: maskBrush, erase: maskErase }]);
  }

  function maskPointerMove(event) {
    if (!maskMode || maskSubTool !== "brush" || !maskPaintingRef.current) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    setMaskLines((prev) => {
      if (!prev.length) return prev;
      const last = prev[prev.length - 1];
      return [...prev.slice(0, -1), { ...last, points: [...last.points, pt.x, pt.y] }];
    });
  }

  function maskPointerUp() {
    maskPaintingRef.current = false;
  }

  function clearMask() {
    setMaskLines([]);
    setMaskBaseImage(null);
    setMaskOverlay(null);
  }

  // ── Smart-select box (sc-3751) ────────────────────────────────────────────
  // A box-drag sub-mode of the mask tool: drag a selection rect, then on release run the SAM3
  // `image_segment` job over it. Mirrors the sc-6090 box draw, but transient (one rect → one run).
  function selectPointerDown(event) {
    if (!maskMode || maskSubTool !== "select" || !working) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    selectDrawingRef.current = true;
    selectStartRef.current = pt;
    setSelectDraft({ x: pt.x, y: pt.y, width: 0, height: 0 });
  }

  function selectPointerMove(event) {
    if (!selectDrawingRef.current) return;
    const pt = stagePointToImage(event);
    if (!pt) return;
    setSelectDraft(rectFromPoints(selectStartRef.current, pt));
  }

  function selectPointerUp() {
    if (!selectDrawingRef.current) return;
    selectDrawingRef.current = false;
    const draft = selectDraft;
    setSelectDraft(null);
    // Discard a click / sub-minimum smudge; otherwise run segmentation over the box.
    if (!draft || draft.width < MIN_BOX_PX || draft.height < MIN_BOX_PX) return;
    const rect = clampRectToCanvas(draft, working.width, working.height);
    runSmartSelect(rect);
  }

  // Cancel an in-flight smart-select drag (called by the editor's escapeGesture).
  // Returns true when it actually cancelled one, so escapeGesture keeps its priority order.
  function cancelSelectDrag() {
    if (!selectDrawingRef.current) return false;
    selectDrawingRef.current = false;
    setSelectDraft(null);
    return true;
  }

  // Rasterize the brush strokes to a mask PNG File aligned to the working bitmap:
  // white = edit region on black. Erase strokes punch holes (destination-out on a
  // transparent scratch), then it's flattened onto black so the worker's convert("L")
  // reads white-on-black. Mirrors the same compositing as the on-canvas preview.
  // Composite the current mask (smart-select base + brush strokes) onto a fresh
  // white-on-black canvas at working dims. Shared by the inpaint upload + the mask
  // refine ops (sc-6110). White = edit region; erased holes flatten to black (=keep).
  function rasterizeMaskToCanvas() {
    const scratch = document.createElement("canvas");
    scratch.width = working.width;
    scratch.height = working.height;
    const sctx = scratch.getContext("2d");
    // Smart-select base first (sc-3751): the white-on-black SAM3 mask underlays the brush strokes,
    // so paint strokes add to it and erase strokes (destination-out) carve it back. Its opaque
    // black areas are harmless — the final flatten is onto black anyway.
    if (maskBaseImage) sctx.drawImage(maskBaseImage, 0, 0);
    sctx.lineCap = "round";
    sctx.lineJoin = "round";
    sctx.strokeStyle = "#ffffff";
    sctx.fillStyle = "#ffffff";
    for (const line of maskLines) {
      sctx.globalCompositeOperation = line.erase ? "destination-out" : "source-over";
      sctx.lineWidth = line.size;
      const p = line.points;
      if (p.length === 2) {
        sctx.beginPath();
        sctx.arc(p[0], p[1], line.size / 2, 0, Math.PI * 2);
        sctx.fill();
        continue;
      }
      sctx.beginPath();
      sctx.moveTo(p[0], p[1]);
      for (let i = 2; i < p.length; i += 2) sctx.lineTo(p[i], p[i + 1]);
      sctx.stroke();
    }
    // Flatten onto black so erased/holes read as black (= keep).
    const out = document.createElement("canvas");
    out.width = working.width;
    out.height = working.height;
    const octx = out.getContext("2d");
    octx.fillStyle = "#000000";
    octx.fillRect(0, 0, out.width, out.height);
    octx.drawImage(scratch, 0, 0);
    return out;
  }

  function rasterizeMaskToFile() {
    return new Promise((resolve, reject) => {
      rasterizeMaskToCanvas().toBlob((blob) => {
        if (!blob) {
          reject(new Error("Could not encode the mask."));
          return;
        }
        resolve(new File([blob], "mask.png", { type: "image/png" }));
      }, "image/png");
    });
  }

  // Decode a worker mask (white-on-black PNG at working dims) into the editable mask base: a
  // white-on-black canvas for rasterizeMaskToFile + a pink-on-transparent overlay for the preview.
  // Drawn scaled to the working dims defensively (the mask is produced at the working size).
  function loadMaskBase(image) {
    const base = document.createElement("canvas");
    base.width = working.width;
    base.height = working.height;
    base.getContext("2d").drawImage(image, 0, 0, working.width, working.height);
    const overlay = document.createElement("canvas");
    overlay.width = working.width;
    overlay.height = working.height;
    const octx = overlay.getContext("2d");
    octx.drawImage(base, 0, 0);
    const data = octx.getImageData(0, 0, overlay.width, overlay.height);
    tintMaskRgbaInPlace(data.data);
    octx.putImageData(data, 0, 0);
    setMaskBaseImage(base);
    setMaskOverlay(overlay);
  }

  // Install a refined white-on-black mask canvas as the new mask base (sc-6110): it
  // becomes the base + a fresh pink overlay, and the brush strokes are cleared (they
  // are now baked into the canvas). Mirrors loadMaskBase but from a canvas.
  function applyRefinedMask(maskCanvas) {
    const overlay = document.createElement("canvas");
    overlay.width = working.width;
    overlay.height = working.height;
    const octx = overlay.getContext("2d");
    octx.drawImage(maskCanvas, 0, 0);
    const data = octx.getImageData(0, 0, overlay.width, overlay.height);
    tintMaskRgbaInPlace(data.data);
    octx.putImageData(data, 0, 0);
    setMaskBaseImage(maskCanvas);
    setMaskOverlay(overlay);
    setMaskLines([]);
  }

  // Mask refinement (sc-6110): flatten the current mask, run a pure pixel op
  // (feather / grow / shrink / invert), and reinstall it as the base. No-op when no
  // mask exists, except invert (empty mask → select-all).
  function refineMask(op) {
    if (!working) return;
    if (op !== "invert" && !maskHasContent(maskLines) && !maskBaseImage) return;
    const canvas = rasterizeMaskToCanvas();
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, w, h);
    const alpha = maskAlphaFromRgba(imageData.data);
    let refined;
    if (op === "invert") refined = invertAlpha(alpha);
    else if (op === "grow") refined = dilateAlpha(alpha, w, h, maskRefineRadius);
    else if (op === "shrink") refined = erodeAlpha(alpha, w, h, maskRefineRadius);
    else refined = blurAlpha(alpha, w, h, maskRefineRadius);
    writeMaskAlphaToRgba(imageData.data, refined);
    ctx.putImageData(imageData, 0, 0);
    applyRefinedMask(canvas);
  }

  // Run the SAM3 image_segment job over the selection box (sc-3751): stage the working image, post
  // the job, and on completion load the returned binary mask as an editable base under the strokes.
  // It does NOT replace the working image (onComplete owns the result), so the session is unchanged
  // except for the mask layer; the brush/eraser then refines it before Inpaint.
  function runSmartSelect(rect) {
    if (!working || aiOp || !canMask) return;
    const box = rectToSegmentBox(rect);
    runAiOp({
      label: "smart select",
      endpoint: "/api/v1/jobs",
      buildBody: (scratch) =>
        buildSegmentJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          box,
          displayName: working?.source?.name,
        }),
      onComplete: async (resultAsset) => {
        const res = await fetch(assetUrl(resultAsset));
        if (!res.ok) throw new Error(`Failed to load mask (${res.status})`);
        const { image, objectUrl } = await blobToImage(await res.blob());
        loadMaskBase(image);
        URL.revokeObjectURL(objectUrl); // the pixels are copied into the base/overlay canvases
        // Return to the mask tool so the auto-mask can be refined with the brush/eraser.
        setTool("edit");
        setMaskMode(true);
        setMaskSubTool("brush");
      },
    });
  }

  return {
    // State
    maskLines,
    maskMode,
    maskBrush,
    maskErase,
    maskRefineRadius,
    maskBaseImage,
    maskOverlay,
    maskSubTool,
    selectDraft,
    // Setters used in render
    setMaskMode,
    setMaskBrush,
    setMaskErase,
    setMaskRefineRadius,
    setMaskSubTool,
    // Gesture-latch ref exposed for escapeGesture / resetEditorOverlays
    selectDrawingRef,
    // Handlers
    maskPointerDown,
    maskPointerMove,
    maskPointerUp,
    clearMask,
    selectPointerDown,
    selectPointerMove,
    selectPointerUp,
    cancelSelectDrag,
    rasterizeMaskToFile,
    refineMask,
    resetMaskState,
  };
}
