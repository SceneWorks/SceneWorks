import { useCallback, useEffect, useState } from "react";
import {
  IDENTITY_LEVELS,
  IDENTITY_CURVES,
  isIdentityLevels,
  isIdentityCurves,
  applyLevels,
  applyCurves,
  computeHistogram,
} from "../../colorGrade.js";
import { activeLayerOf } from "../../imageLayers.js";
import {
  COLOR_ADJUSTMENTS,
  IDENTITY_COLOR_ADJUST,
  isIdentityAdjust,
  applyColorAdjustments,
} from "./colorGradeMath.js";

// Color grade tool (sc-2439; curves + levels sc-6109) extracted from ImageEditor.jsx
// (sc-9752, F-052 follow-up). Owns the non-destructive −1..1 adjust / levels / curves
// state, the live-preview + histogram effects, and the Apply bake. Behavior-preserving:
// the state, the effect bodies, and the effect dependency arrays (including the two
// `eslint-disable-next-line react-hooks/exhaustive-deps` on the preview + histogram
// effects) are byte-identical to the pre-extraction inline versions.
//
// There is NO ref-mirror state here — the color grade is plain React state read only in
// render + the two effects + the Apply callback, so nothing is mirrored into a ref for a
// live pointer/canvas read (unlike the mask + boxes tools). The color state does feed the
// component's undo snapshot ONLY indirectly: a grade is baked (Apply) into the active
// layer before a checkpoint, so the snapshot captures the baked bitmap, not the live
// slider values — this hook keeps that exact contract.
//
// `working`, `tool`, and `imageNodeRef` are read live inside the effects; the caller
// feeds them from render scope so the effects re-run on exactly the same inputs as before.
export function useColorGradeTool({
  working,
  tool,
  imageNodeRef,
  histogramRef,
  checkpoint,
  replaceLayerImage,
  blobToImage,
  setTool,
  setEdits,
  setDirty,
}) {
  // Color grade (sc-2439): non-destructive −1..1 adjustments previewed live via a
  // Konva filter, baked into the working image on Apply.
  const [colorAdjust, setColorAdjust] = useState(IDENTITY_COLOR_ADJUST);
  // Curves + levels (sc-6109): the Color tool has three modes — the brightness/
  // contrast "adjust" (above), per-channel levels, and an editable tone curve. Each
  // previews via the same Konva filter and bakes via the same Canvas-2D pass. The
  // active channel ("master" | "r" | "g" | "b") is shared by the levels + curves UI.
  const [colorMode, setColorMode] = useState("adjust"); // "adjust" | "levels" | "curves"
  const [levels, setLevels] = useState(IDENTITY_LEVELS);
  const [curves, setCurves] = useState(IDENTITY_CURVES);
  const [colorChannel, setColorChannel] = useState("master");

  // Reset the per-bitmap color-preview state (called by resetEditorOverlays + cancel).
  const resetColorState = useCallback(() => {
    setColorAdjust(IDENTITY_COLOR_ADJUST);
    setColorMode("adjust");
    setLevels(IDENTITY_LEVELS);
    setColorChannel("master");
    setCurves(IDENTITY_CURVES);
  }, []);

  // Discard any unbaked color preview (adjust / levels / curves) WITHOUT touching mode /
  // channel — matches cancelCrop's original behavior exactly (it reset only the three
  // grade values, leaving colorMode + colorChannel as they were).
  const discardColorPreview = useCallback(() => {
    setColorAdjust(IDENTITY_COLOR_ADJUST);
    setLevels(IDENTITY_LEVELS);
    setCurves(IDENTITY_CURVES);
  }, []);

  function startColorGrade() {
    if (!working) return;
    setTool("color");
    setColorMode("adjust");
    setColorChannel("master");
    setColorAdjust(IDENTITY_COLOR_ADJUST);
    setLevels(IDENTITY_LEVELS);
    setCurves(IDENTITY_CURVES);
  }

  const setAdjustValue = (key, value) => setColorAdjust((prev) => ({ ...prev, [key]: value }));
  const resetAdjust = (key) => setAdjustValue(key, 0);

  // Patch the active channel's levels (sc-6109).
  const setLevelsValue = (key, value) =>
    setLevels((prev) => ({ ...prev, [colorChannel]: { ...prev[colorChannel], [key]: value } }));

  // Reset the currently-selected color mode (all channels) to its identity.
  function resetActiveColorMode() {
    if (colorMode === "levels") setLevels(IDENTITY_LEVELS);
    else if (colorMode === "curves") setCurves(IDENTITY_CURVES);
    else setColorAdjust(IDENTITY_COLOR_ADJUST);
  }

  // Stroke for the curve editor / channel cue.
  const channelStroke = { master: "var(--accent)", r: "#d44", g: "#4a4", b: "#46d" }[colorChannel];

  // Whether the currently-selected color mode is at its identity (gates Apply + the
  // live-preview cache).
  function activeGradeIsIdentity() {
    if (colorMode === "levels") return isIdentityLevels(levels);
    if (colorMode === "curves") return isIdentityCurves(curves);
    return isIdentityAdjust(colorAdjust);
  }

  // Live preview: Konva applies filters only on a cached node, and re-running them
  // needs a re-cache. Cache the active layer's node (re-caching when ANY grade input
  // changes) while the color tool is active with a non-identity grade; clear it
  // otherwise so Move/other tools see the untouched bitmap. The filter reads the
  // gradeMode + colorAdjust/levels/curves attrs.
  useEffect(() => {
    const node = imageNodeRef.current;
    if (!node) return;
    if (tool === "color" && !activeGradeIsIdentity()) {
      node.cache();
    } else {
      node.clearCache();
    }
    node.getLayer()?.batchDraw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool, colorMode, colorAdjust, levels, curves, working]);

  // Draw the active layer's histogram for the levels mode (sc-6109). Recomputed when
  // the layer or selected channel changes; cheap (one pass over the layer bitmap).
  useEffect(() => {
    const canvas = histogramRef.current;
    if (!canvas || tool !== "color" || colorMode !== "levels") return;
    const layer = activeLayerOf(working);
    if (!layer) return;
    const off = document.createElement("canvas");
    off.width = layer.image.naturalWidth;
    off.height = layer.image.naturalHeight;
    const octx = off.getContext("2d");
    octx.drawImage(layer.image, 0, 0);
    const hist = computeHistogram(octx.getImageData(0, 0, off.width, off.height).data);
    const series = colorChannel === "master" ? hist.luma : hist[colorChannel];
    const peak = Math.max(1, ...series);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = colorChannel === "r" ? "#d44" : colorChannel === "g" ? "#4a4" : colorChannel === "b" ? "#46d" : "#888";
    const bw = canvas.width / 256;
    for (let i = 0; i < 256; i += 1) {
      const h = (series[i] / peak) * canvas.height;
      ctx.fillRect(i * bw, canvas.height - h, Math.max(1, bw), h);
    }
    // Deps mirror the pre-extraction inline effect exactly; `histogramRef` is a stable ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool, colorMode, colorChannel, working]);

  // Apply: bake the active mode's grade (adjust / levels / curves) into the ACTIVE
  // layer using the SAME pixel math as the live preview (a 2D-canvas pass), writing
  // it back in place — stack + dims preserved (sc-6119). Records the grade in the
  // edit chain (sc-6109).
  const applyColorGrade = useCallback(async () => {
    const layer = activeLayerOf(working);
    if (!working || !layer) return;
    // Resolve the active mode's transform + provenance entry; bail if it's identity.
    let transform;
    let edit;
    if (colorMode === "levels") {
      if (isIdentityLevels(levels)) return;
      const baked = levels;
      transform = (data) => applyLevels(data, baked);
      edit = { op: "levels", levels: baked };
    } else if (colorMode === "curves") {
      if (isIdentityCurves(curves)) return;
      const baked = curves;
      transform = (data) => applyCurves(data, baked);
      edit = { op: "curves", curves: baked };
    } else {
      if (isIdentityAdjust(colorAdjust)) return;
      const baked = { ...colorAdjust };
      transform = (data) => applyColorAdjustments(data, baked);
      edit = { op: "color", ...baked };
    }
    const w = layer.image.naturalWidth;
    const h = layer.image.naturalHeight;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(layer.image, 0, 0);
    const imageData = ctx.getImageData(0, 0, w, h);
    transform(imageData.data);
    ctx.putImageData(imageData, 0, 0);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) return;
    const { image, objectUrl } = await blobToImage(blob);
    checkpoint();
    replaceLayerImage(layer.id, image, objectUrl, blob);
    // The grade is baked; drop every live preview.
    setColorAdjust(IDENTITY_COLOR_ADJUST);
    setLevels(IDENTITY_LEVELS);
    setCurves(IDENTITY_CURVES);
    setTool("move");
    setEdits((prev) => [...prev, edit]);
    setDirty(true);
  }, [working, colorMode, colorAdjust, levels, curves, replaceLayerImage, checkpoint]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    // State (read in render)
    colorAdjust,
    colorMode,
    levels,
    curves,
    colorChannel,
    // Setters used directly in render (mode/channel/curves patch)
    setColorMode,
    setColorChannel,
    setCurves,
    // Derived + handlers
    channelStroke,
    activeGradeIsIdentity,
    startColorGrade,
    setAdjustValue,
    resetAdjust,
    setLevelsValue,
    resetActiveColorMode,
    applyColorGrade,
    // Reset hooks for the component's shared reset/cancel plumbing
    resetColorState,
    discardColorPreview,
    // Pure data re-exported for the render (the adjust slider list)
    COLOR_ADJUSTMENTS,
  };
}
