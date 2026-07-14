import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LoraKeywordSummary } from "../components/LoraKeywordSummary.jsx";
import { createPortal } from "react-dom";
import { Stage, Layer, Image as KonvaImage, Line, Rect, Transformer } from "react-konva";
import { apiFetch } from "../api.js";
import { terminalStatuses } from "../jobTypes.js";
import { useAppContext } from "../context/AppContext.js";
import { useScreenActive } from "../context/ScreenActiveContext.js";
import { appConfirm } from "../appConfirm.jsx";
import { DEFAULT_MAC_CAPABILITIES, macFeatureBlock } from "../macGating.js";
import { assetUrl, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { DatasetAddDialog } from "../components/DatasetAddDialog.jsx";
import { FitModeControl, effectiveFitMode } from "../components/FitModeControl.jsx";
import { useLoraSelection } from "../components/LoraPickerField.jsx";
import {
  LORA_WEIGHT_MAX,
  LORA_WEIGHT_MIN,
  LORA_WEIGHT_STEP,
  MAX_JOB_LORAS_TOTAL,
  findModelEditLora,
  loraIsInstalled,
} from "../presetUtils.js";
import { guidanceDefaultFromModel } from "../samplerOptions.js";
import {
  BLEND_MODES,
  activeLayerOf,
  addLayer,
  compositeLayersToCanvas,
  createLayer,
  duplicateLayer,
  identityTransform,
  layerById,
  moveLayer,
  removeLayer,
  replaceLayerBitmap,
  sameLayerStack,
  setActiveLayer,
  setLayerProps,
  singleLayerWorking,
  snapshotLayers,
} from "../imageLayers.js";
import { CurveEditor } from "../components/CurveEditor.jsx";
// Pure job builders + model/engine helpers (sc-6112) — extracted to a konva-free module
// so the Library batch flow can reuse them; imported for internal use and re-exported
// below to keep this module's public surface (and its tests) unchanged.
import {
  buildDetailJobBody,
  buildEditJobBody,
  buildUpscaleJobBody,
  detailCapableModels,
  editCapableModels,
  tileControlNetInstalled,
  tileControlNetModel,
  TILE_CONTROLNET_MODEL_ID,
  upscaleEngineHasSoftness,
  upscaleFactorsForEngine,
} from "../imageJobs.js";
import {
  availableUpscaleEngines as upscaleEnginesForPlatform,
  useUpscaleEngineFallback,
} from "../upscaleEngines.js";
// Per-tool logic extracted to dedicated hooks + pure helper modules (sc-9752, F-052
// follow-up). Each hook owns its tool's state/refs/handlers and is wired below; the pure
// helpers are re-exported here so this module's public surface (and ImageEditor.test.jsx's
// imports) stay byte-for-byte unchanged.
import { useColorGradeTool } from "./imageEditor/useColorGradeTool.js";
import { useBoxesTool } from "./imageEditor/useBoxesTool.js";
import { useMaskTool } from "./imageEditor/useMaskTool.js";
import {
  COLOR_ADJUSTMENTS,
  IDENTITY_COLOR_ADJUST,
  isIdentityAdjust,
  gradePixel,
  applyColorAdjustments,
  konvaColorFilter,
} from "./imageEditor/colorGradeMath.js";
import {
  BOX_TYPES,
  MAX_BOX_PALETTE,
  MAX_DOCUMENT_PALETTE,
  isValidHexColor,
  rectToBbox,
  bboxToRect,
  boxPaletteIsValid,
  documentPalette,
  documentPaletteIsValid,
  boxIsValid,
  BOX_PALETTE,
  MIN_BOX_PX,
  rectFromPoints,
  clampRectToCanvas,
  makeBox,
  boxFillStyle,
  addPaletteColor,
  removePaletteColor,
  boxMetadataGaps,
  paintBoxesOnContext,
  colorName,
  composeColorPrompt,
  boxesToIdeogramElements,
} from "./imageEditor/boxGeometry.js";
import {
  buildSegmentJobBody,
  rectToSegmentBox,
  tintMaskRgbaInPlace,
  MASK_PREVIEW_RGBA,
  maskHasContent,
} from "./imageEditor/maskShared.js";
import { loraAddHint } from "./imageEditor/loraSection.js";

export {
  buildDetailJobBody,
  buildEditJobBody,
  buildUpscaleJobBody,
  detailCapableModels,
  editCapableModels,
  tileControlNetInstalled,
  tileControlNetModel,
  TILE_CONTROLNET_MODEL_ID,
  upscaleEngineHasSoftness,
  upscaleFactorsForEngine,
};

// Re-export the per-tool pure helpers (sc-9752) so the editor's public surface + its
// test imports are unchanged after the extraction.
export {
  COLOR_ADJUSTMENTS,
  IDENTITY_COLOR_ADJUST,
  isIdentityAdjust,
  gradePixel,
  applyColorAdjustments,
  konvaColorFilter,
  BOX_TYPES,
  MAX_BOX_PALETTE,
  MAX_DOCUMENT_PALETTE,
  isValidHexColor,
  rectToBbox,
  bboxToRect,
  boxPaletteIsValid,
  documentPalette,
  documentPaletteIsValid,
  boxIsValid,
  BOX_PALETTE,
  MIN_BOX_PX,
  rectFromPoints,
  clampRectToCanvas,
  makeBox,
  boxFillStyle,
  addPaletteColor,
  removePaletteColor,
  boxMetadataGaps,
  paintBoxesOnContext,
  colorName,
  composeColorPrompt,
  boxesToIdeogramElements,
  buildSegmentJobBody,
  rectToSegmentBox,
  tintMaskRgbaInPlace,
  MASK_PREVIEW_RGBA,
  maskHasContent,
};

const MIN_SCALE = 0.05;
const MAX_SCALE = 16;
const ZOOM_STEP = 1.2;
const MIN_CROP_PX = 8;

// Redesign panel layout (epic 10243): accordion (default) / right / left / bottom,
// persisted across sessions. Invalid/absent → accordion.
export const EDITOR_LAYOUTS = ["accordion", "right", "left", "bottom"];
const EDITOR_LAYOUT_KEY = "sceneworks-ie-layout";
export function readStoredEditorLayout() {
  try {
    const saved = window.localStorage.getItem(EDITOR_LAYOUT_KEY);
    if (EDITOR_LAYOUTS.includes(saved)) return saved;
  } catch {
    /* ignore (private mode etc.) */
  }
  return "accordion";
}

// Tool identity for the rail / accordion headers / inspector header (epic 10243).
export const EDITOR_TOOL_ORDER = ["move", "transform", "crop", "upscale", "detail", "color", "edit", "boxes"];
export const EDITOR_TOOL_META = {
  move: { label: "Move", desc: "Pan and inspect the canvas" },
  transform: { label: "Transform", desc: "Move, scale & rotate the layer" },
  crop: { label: "Crop", desc: "Trim to a size or ratio" },
  upscale: { label: "Upscale", desc: "Increase resolution with AI" },
  detail: { label: "Detail", desc: "Refine texture with tile ControlNet" },
  color: { label: "Color grade", desc: "Adjust tone, levels & curves" },
  edit: { label: "AI Edit", desc: "Prompt-driven edit & inpaint" },
  boxes: { label: "Boxes", desc: "Region layout & color-keyed edit" },
};

// Inline stroke icons ported from the design handoff `ICONS` map (epic 10243).
const strokeSvg = (props, children) => (
  <svg fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.9} viewBox="0 0 24 24" {...props}>
    {children}
  </svg>
);
export const EDITOR_TOOL_ICONS = {
  move: strokeSvg({ key: "move" }, [
    <polyline key="a" points="5 9 2 12 5 15" />, <polyline key="b" points="9 5 12 2 15 5" />,
    <polyline key="c" points="15 19 12 22 9 19" />, <polyline key="d" points="19 9 22 12 19 15" />,
    <line key="e" x1="2" x2="22" y1="12" y2="12" />, <line key="f" x1="12" x2="12" y1="2" y2="22" />,
  ]),
  transform: strokeSvg({ key: "transform" }, [
    <rect key="a" height="16" rx="1" width="16" x="4" y="4" />,
    <circle key="b" cx="4" cy="4" fill="currentColor" r="2" />, <circle key="c" cx="20" cy="4" fill="currentColor" r="2" />,
    <circle key="d" cx="4" cy="20" fill="currentColor" r="2" />, <circle key="e" cx="20" cy="20" fill="currentColor" r="2" />,
  ]),
  crop: strokeSvg({ key: "crop" }, [
    <path key="a" d="M6 2v14a2 2 0 0 0 2 2h14" />, <path key="b" d="M18 22V8a2 2 0 0 0-2-2H2" />,
  ]),
  upscale: strokeSvg({ key: "upscale" }, [
    <polyline key="a" points="15 3 21 3 21 9" />, <polyline key="b" points="9 21 3 21 3 15" />,
    <line key="c" x1="21" x2="14" y1="3" y2="10" />, <line key="d" x1="3" x2="10" y1="21" y2="14" />,
  ]),
  detail: strokeSvg({ key: "detail" }, [
    <path key="a" d="M12 3l1.9 4.8L19 9.5l-4.1 2.9L16 18l-4-2.7L8 18l1.1-5.6L5 9.5l5.1-1.7z" />,
  ]),
  color: strokeSvg({ key: "color" }, [
    <circle key="a" cx="12" cy="12" r="9" />, <path key="b" d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" stroke="none" />,
  ]),
  edit: strokeSvg({ key: "edit" }, [
    <path key="a" d="M15 4l5 5" />, <path key="b" d="M4 20l4-1 10-10-3-3L5 16z" />,
    <path key="c" d="M14 5l1.5-1.5a2 2 0 0 1 3 3L17 8" />,
  ]),
  boxes: strokeSvg({ key: "boxes" }, [
    <rect key="a" height="7" rx="1" width="8" x="3" y="4" />, <rect key="b" height="12" rx="1" width="8" x="13" y="4" />,
    <rect key="c" height="6" rx="1" width="8" x="3" y="14" />,
  ]),
};
const IeChevron = () => (
  <svg fill="none" height="15" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.2} viewBox="0 0 24 24" width="15">
    <path d="M6 9l6 6 6-6" />
  </svg>
);
const IeEyeOpen = () => (
  <svg fill="none" height="15" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24" width="15">
    <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);
const IeEyeOff = () => (
  <svg className="ie-vis-off" fill="none" height="15" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24" width="15">
    <path d="M17.9 17.9A10.4 10.4 0 0 1 12 19C5 19 1 12 1 12a19 19 0 0 1 5.1-5.9M9.9 4.2A10.9 10.9 0 0 1 12 4c7 0 11 7 11 7a19 19 0 0 1-2.2 3.2M1 1l22 22" />
  </svg>
);

// Modifier glyph for the shortcut reference (the handler accepts both ⌘ and Ctrl).
const IS_MAC =
  typeof navigator !== "undefined" && /Mac|iP(hone|ad|od)/.test(navigator.platform || navigator.userAgent || "");
const MOD_KEY = IS_MAC ? "⌘" : "Ctrl";

// Keyboard shortcut reference (sc-6111). Single source of truth for the in-editor
// quick reference; the keydown handler implements these exact bindings.
const EDITOR_SHORTCUTS = [
  {
    group: "Tools",
    items: [
      { keys: ["M"], label: "Move / pan" },
      { keys: ["T"], label: "Transform layer" },
      { keys: ["C"], label: "Crop" },
      { keys: ["U"], label: "Upscale" },
      { keys: ["D"], label: "Detail enhance" },
      { keys: ["G"], label: "Color grade" },
      { keys: ["E"], label: "AI edit" },
      { keys: ["B"], label: "Boxes" },
    ],
  },
  {
    group: "View",
    items: [
      { keys: ["+"], label: "Zoom in" },
      { keys: ["−"], label: "Zoom out" },
      { keys: ["0"], label: "Fit to view" },
      { keys: ["1"], label: "Actual size (100%)" },
    ],
  },
  {
    group: "Edit",
    items: [
      { keys: [MOD_KEY, "Z"], label: "Undo" },
      { keys: ["⇧", MOD_KEY, "Z"], label: "Redo" },
      { keys: ["Delete"], label: "Delete selected box" },
      { keys: ["Esc"], label: "Cancel / deselect" },
      { keys: ["?"], label: "Toggle this help" },
    ],
  },
];

// Upper bound on images in a FLUX.2 multi-reference edit, matching the worker's MAX_EDIT_REFERENCES
// (image_jobs/flux2.rs). The working image takes one slot, so the editor allows up to 3 user refs.
export const MAX_EDIT_REFERENCES = 4;

// The leave-guard prompt shown before navigating away from the editor, or null when it's
// safe to leave silently (sc-2434 / sc-8850). Unsaved edits win the message. Crucially the
// guard also fires while an AI op is in flight — starting one does NOT set `dirty` (its
// result only lands on success), so without this branch the user could silently abandon a
// running edit (losing the result and, before the App survivor sweep, orphaning scratch).
export function leaveGuardMessage({ dirty, aiOpPending }) {
  if (dirty) return "You have unsaved edits in the Image Editor. Leave and discard them?";
  if (aiOpPending) return "An image edit is still running. Leave and abandon it?";
  return null;
}

// Decide which of the two leave-guards to arm (sc-11959). The browser-unload
// (close/refresh) guard must arm whenever there are unsaved edits or an in-flight AI op,
// EVEN when the editor is backgrounded under keep-alive — an app close/refresh would
// otherwise silently discard a backgrounded editor's unsaved edits (a safeguard that
// predates keep-alive). The in-app nav guard, by contrast, only arms while the editor is
// the foreground view, so a backgrounded (still-mounted) editor doesn't re-fire its
// confirm() on every UNRELATED navigation between other screens.
export function leaveGuardArming({ dirty, aiOpPending, screenActive }) {
  const message = leaveGuardMessage({ dirty, aiOpPending });
  return {
    message,
    beforeUnload: Boolean(message),
    inApp: Boolean(message) && Boolean(screenActive),
  };
}

// The confirm prompt shown when the user EXPLICITLY closes/discards the working image
// (the Close button, sc-11968). Unlike the passive leave guard, a deliberate close of a
// CLEAN document loses nothing, so it needs no confirm (null → proceed silently). Unsaved
// edits win the wording; an in-flight AI op is the fallback (its result would be abandoned).
export function closeConfirmMessage({ dirty, aiOpPending }) {
  if (dirty) return "Discard your unsaved edits and close this image?";
  if (aiOpPending) return "An image edit is still running. Close and abandon it?";
  return null;
}

// Which save-state indicator the top bar shows (sc-11968): the unsaved-edits pill while
// `dirty`, the "Saved ✓" hint once a Save has landed and nothing has changed since, else
// nothing. Pure so the badge logic is unit-testable without a mounted canvas.
export function saveStatusIndicator({ dirty, savedAssetId }) {
  if (dirty) return "unsaved";
  if (savedAssetId) return "saved";
  return null;
}

// Compose the multi-reference edit's `referenceAssetIds`: the working image (staged as the scratch
// source) FIRST so it anchors the edit, then the user's reference images — trimmed, de-duped, and
// capped at `max` total. The worker prefers a non-empty `referenceAssetIds` list over `sourceAssetId`
// (image_jobs/flux2.rs `flux2_edit_reference_ids`), so the working scratch id must lead the list or
// the working image is dropped from the joint conditioning. Pure. Empty source → empty list.
export function editReferenceIds(sourceAssetId, refIds, max = MAX_EDIT_REFERENCES) {
  const ids = [sourceAssetId, ...(refIds ?? [])]
    .map((id) => (typeof id === "string" ? id.trim() : ""))
    .filter(Boolean);
  return Array.from(new Set(ids)).slice(0, max);
}

// Output aspect presets for the editor's canvas-extend / outpaint control (sc-2556).
// "match" keeps the working size, so the fit mode then has no border to act on.
export const EDIT_OUTPUT_ASPECTS = [
  { key: "match", label: "Match canvas", ratio: null },
  { key: "1:1", label: "1:1", ratio: 1 },
  { key: "16:9", label: "16:9", ratio: 16 / 9 },
  { key: "9:16", label: "9:16", ratio: 9 / 16 },
  { key: "4:3", label: "4:3", ratio: 4 / 3 },
  { key: "3:4", label: "3:4", ratio: 3 / 4 },
  { key: "3:2", label: "3:2", ratio: 3 / 2 },
  { key: "2:3", label: "2:3", ratio: 2 / 3 },
];

export function editOutputAspectRatio(key) {
  return EDIT_OUTPUT_ASPECTS.find((aspect) => aspect.key === key)?.ratio ?? null;
}

// Snap an edit-output pixel dimension to a multiple of 16 (min 16). Every image engine
// requires dims divisible by 16 (VAE ×8 · patch ×2 — e.g. mlx-gen z-image's SIZE_MULTIPLE
// guard); an imported/cropped source at arbitrary dims (e.g. 832×1165) would otherwise be
// forwarded verbatim and hard-fail at generation. Mirrors snapCanvasDim's rounding, without
// its blank-canvas [256, 2048] clamp — an edit must not force-grow a small source to 256.
function alignEditDim(px) {
  return Math.max(16, Math.round(px / 16) * 16);
}

// Output W×H for an editor edit given the target aspect + fit mode, keeping the working
// image at native scale (never upscales). "match"/unknown aspect → working size. crop =
// largest target-aspect rect INSIDE the image (trim the overflow); pad/outpaint =
// smallest target-aspect canvas CONTAINING the image (extend → border to fill). Result dims
// are always snapped to a multiple of 16 so the engine's size guard accepts them; the worker
// crop/pad-fits the source to these dims (never stretches). Pure.
export function editOutputDims(workingW, workingH, aspectKey, fitMode) {
  const ratio = editOutputAspectRatio(aspectKey);
  if (!ratio || !workingW || !workingH) {
    return { width: alignEditDim(workingW), height: alignEditDim(workingH) };
  }
  const imageRatio = workingW / workingH;
  let width;
  let height;
  if (fitMode === "crop") {
    // Cover: shrink to the target aspect within the image (trim).
    if (ratio >= imageRatio) {
      width = workingW;
      height = Math.round(workingW / ratio);
    } else {
      height = workingH;
      width = Math.round(workingH * ratio);
    }
  } else {
    // Pad / outpaint: extend to the target aspect around the image (add border).
    if (ratio >= imageRatio) {
      height = workingH;
      width = Math.round(workingH * ratio);
    } else {
      width = workingW;
      height = Math.round(workingW / ratio);
    }
  }
  return { width: alignEditDim(width), height: alignEditDim(height) };
}

// Whether a model accepts an inpaint mask — the manifest tags it `image_inpaint`
// (sc-2476). Gates the mask tool in the editor. Pure.
export function modelIsInpaintCapable(model) {
  return (model?.capabilities ?? []).includes("image_inpaint");
}

// Filename for a Save / Download export (sc-2434): the source name with an
// "-edited" suffix before the extension, always .png — the working image is
// rasterized to PNG, so the original extension would be misleading. Pure.
export function editedFilename(source) {
  const base = (source?.name || "image").replace(/\.[^./\\]+$/, "").trim() || "image";
  return `${base}-edited.png`;
}

// Provenance for a saved edit, stored under the new asset's top-level `extra`
// (sc-2434): which source it was derived from + the ordered edit chain
// (crop/upscale/…) applied this session. Pure for unit testing.
export function buildSaveProvenance({ source, edits, width, height, layers }) {
  const provenance = {
    editor: "image_editor",
    source: source?.assetId
      ? { kind: "asset", assetId: source.assetId, name: source.name ?? null }
      : { kind: "upload", name: source?.name ?? null },
    edits: edits ?? [],
    width: width ?? null,
    height: height ?? null,
  };
  // Layer summary (sc-6121): record what the flattened asset was composited from —
  // bottom→top name / opacity / blend / visibility. Omitted for the degenerate
  // single-layer document (it adds nothing over the flat bitmap, and keeps a plain
  // non-layered save's provenance byte-for-byte as it was before layers).
  if (Array.isArray(layers) && layers.length > 1) {
    provenance.layers = layers.map((layer) => ({
      name: layer.name,
      opacity: layer.opacity,
      blendMode: layer.blendMode,
      visible: layer.visible,
    }));
  }
  return provenance;
}

// Predefined crop ratios (width / height). Rotate swaps to the transpose; 1:1 and
// Freeform are unaffected.
const CROP_RATIOS = [
  { key: "free", label: "Freeform", ratio: null },
  { key: "1:1", label: "1:1", ratio: 1 },
  { key: "3:4", label: "3:4", ratio: 3 / 4 },
  { key: "5:7", label: "5:7", ratio: 5 / 7 },
  { key: "8:10", label: "8:10", ratio: 8 / 10 },
  { key: "16:9", label: "16:9", ratio: 16 / 9 },
];

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

// Resolve a ratio key (+ rotate) to a concrete width/height ratio, or null for
// freeform. Rotating transposes non-square ratios (3:4 → 4:3); 1:1 is a no-op.
export function cropRatioForKey(key, rotated) {
  const found = CROP_RATIOS.find((entry) => entry.key === key);
  const base = found ? found.ratio : null;
  if (base == null || base === 1) return base;
  return rotated ? 1 / base : base;
}

// Largest rect of the given ratio that fits in the image, centered. Freeform
// (null ratio) defaults to a centered 80% box. Returns image-pixel coords.
export function centeredCropRect(imgW, imgH, ratio) {
  if (ratio == null) {
    const w = imgW * 0.8;
    const h = imgH * 0.8;
    return { x: (imgW - w) / 2, y: (imgH - h) / 2, width: w, height: h };
  }
  let w = imgW;
  let h = w / ratio;
  if (h > imgH) {
    h = imgH;
    w = h * ratio;
  }
  return { x: (imgW - w) / 2, y: (imgH - h) / 2, width: w, height: h };
}

// The four dim rectangles that mask everything outside the crop rect (image coords).
function cropOverlayRects(imgW, imgH, rect) {
  const right = rect.x + rect.width;
  const bottom = rect.y + rect.height;
  return [
    { x: 0, y: 0, width: imgW, height: rect.y },
    { x: 0, y: bottom, width: imgW, height: imgH - bottom },
    { x: 0, y: rect.y, width: rect.x, height: rect.height },
    { x: right, y: rect.y, width: imgW - right, height: rect.height },
  ];
}

// ── Blank-canvas "New layout" (Workstream A, sc-6092) ────────────────────────
// A from-scratch substrate for layout-from-nothing (Ideogram text-to-image). The
// dimensions obey Ideogram's constraints: multiples of 16 within [256, 2048].
export const BLANK_CANVAS_MIN = 256;
export const BLANK_CANVAS_MAX = 2048;
export const BLANK_CANVAS_SIZES = [512, 768, 1024, 1536, 2048];

// Snap a pixel dimension to a multiple of 16 within [256, 2048] (Ideogram limits).
function snapCanvasDim(px) {
  return clamp(Math.round(px / 16) * 16, BLANK_CANVAS_MIN, BLANK_CANVAS_MAX);
}

// Target W×H for a blank layout from an aspect preset + a long-side size. Both
// dims are multiples of 16 in [256, 2048]. "match"/unknown aspect → square. Pure.
export function blankCanvasDims(aspectKey, longSide) {
  const ratio = editOutputAspectRatio(aspectKey) ?? 1;
  let width;
  let height;
  if (ratio >= 1) {
    width = longSide;
    height = longSide / ratio;
  } else {
    height = longSide;
    width = longSide * ratio;
  }
  return { width: snapCanvasDim(width), height: snapCanvasDim(height) };
}

// Decode a blob into an HTMLImageElement via a same-origin object: URL. Asset
// files are served cross-origin from the API in local dev, so loading the bytes
// this way (rather than an <img crossOrigin> against the file URL) guarantees the
// Konva canvas is never tainted — later crop/export (sc-2430/sc-2434) need to read
// pixels back. Resolves { image, objectUrl }; caller owns revoking objectUrl.
function blobToImage(blob) {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => resolve({ image, objectUrl });
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Could not decode image"));
    };
    image.src = objectUrl;
  });
}

// ── Undo/redo history (sc-6106) ────────────────────────────────────────────
// A bounded, backend-free history over opaque working-image snapshots. The
// reducer is pure — it only shuffles snapshots between the past (undo) and future
// (redo) stacks; the caller owns capturing a snapshot (the working bitmap blob +
// box/provenance overlay state) and restoring one (decode + install). Snapshots
// hold a Blob, never a live object URL, so an evicted snapshot is plain garbage —
// there is nothing to revoke, which keeps the "no leak of evicted snapshots"
// guarantee trivial. The stack depth is bounded so retained bitmaps stay capped.
export const HISTORY_LIMIT = 30;

export function emptyHistory() {
  return { past: [], future: [] };
}

// Push the current snapshot onto the undo stack and drop the redo stack. Call at
// the START of an operation, before the working state mutates, with a snapshot of
// the pre-operation state. Bounded to the `limit` most-recent entries.
export function historyCheckpoint(history, snapshot, limit = HISTORY_LIMIT) {
  return { past: [...history.past, snapshot].slice(-limit), future: [] };
}

// Step back one operation. `present` is the current on-screen snapshot, captured
// fresh by the caller so a later redo restores exactly what is on screen now.
// Returns the next history plus the snapshot to restore (`restore` is null when
// there is nothing to undo, in which case `history` is returned unchanged).
export function historyUndo(history, present, limit = HISTORY_LIMIT) {
  if (!history.past.length) return { history, restore: null };
  const restore = history.past[history.past.length - 1];
  return {
    history: { past: history.past.slice(0, -1), future: [present, ...history.future].slice(0, limit) },
    restore,
  };
}

// Step forward one operation, symmetric to historyUndo.
export function historyRedo(history, present, limit = HISTORY_LIMIT) {
  if (!history.future.length) return { history, restore: null };
  const [restore, ...rest] = history.future;
  return {
    history: { past: [...history.past, present].slice(-limit), future: rest },
    restore,
  };
}

export const canUndo = (history) => history.past.length > 0;
export const canRedo = (history) => history.future.length > 0;

// Serialize a single undo/redo step so a rapid second invocation can't race a
// restore that is still in flight (sc-8852). Restoring is async — each changed
// layer is decoded via `blobToImage` before the working stack is re-installed —
// so without a guard a key-repeat Cmd-Z would run `step()` again while the first
// restore is mid-flight, capturing the STALE working state as "present" and
// pushing a duplicate onto the redo (`future`) branch. Both undo AND redo route
// through here and share one `guardRef`: while a restore is in flight, any
// further undo/redo is ignored (the finding's simplest fix — holding Cmd-Z steps
// back predictably rather than dropping-then-jumping). The guard is set BEFORE
// the reducer step runs and cleared in a `finally`, so a mid-restore error can't
// wedge undo/redo forever. Returns true when a step actually ran.
//   - guardRef:     a mutable ref-like `{ current: boolean }` shared by undo/redo.
//   - step:         () => ({ history, restore }) — the pure reducer step, called
//                   only once we hold the guard, capturing the live "present".
//   - commitHistory:(nextHistory) => void — install the new history + sync flags.
//   - restore:      async (snapshot) => void — the async snapshot re-install.
export async function runGuardedRestore({ guardRef, step, commitHistory, restore }) {
  if (guardRef.current) return false;
  guardRef.current = true;
  try {
    const { history: next, restore: target } = step();
    if (!target) return false;
    commitHistory(next);
    await restore(target);
    return true;
  } finally {
    guardRef.current = false;
  }
}

// Revoke the object URLs of a set of live layers (sc-6117). Undo snapshots hold
// only blobs (no URLs), so the only URLs that ever need revoking are the live
// ones, when their layer is evicted — on delete, on a session replace, and on
// unmount. Tolerant of null/missing URLs so callers don't have to guard.
function revokeLayerUrls(layers) {
  for (const layer of layers ?? []) {
    if (layer?.objectUrl) URL.revokeObjectURL(layer.objectUrl);
  }
}

export function ImageEditor() {
  const {
    activeProject,
    assets,
    characters,
    setPreviewAsset,
    token,
    requestedGpu,
    jobs,
    importAsset,
    purgeAsset,
    registerLeaveGuard,
    // App-level scratch-op survivor coordination (sc-8850). The editor stages an ephemeral
    // scratch asset per AI op; these let App purge it (and the result) even if the user
    // navigates away mid-job and this component unmounts before its own watcher can run.
    trackEditorScratchOp,
    releaseEditorScratchOp,
    registerEditorScratchClaim,
    imageModels,
    // Full catalog (all types incl. utility) + downloader — the Detail tool's tile ControlNet is a
    // `type:"utility"` entry, so it is absent from the image-only `imageModels`; we look it up here to
    // gate the run and offer a one-click install (sc-2437/sc-2438 provisioning gap).
    models = [],
    createModelDownloadJob,
    editorLaunch = null,
    clearEditorLaunch,
    macCapabilities = DEFAULT_MAC_CAPABILITIES,
    // Project LoRA catalog (sc-10254): fed to the AI Edit LoRA picker, gated to the
    // edit model's compatible families.
    loras = [],
    // Managed image-edit LoRA download (epic 10871, sc-11069): a missing Krea edit LoRA
    // offers a one-click fetch, mirroring the Image Studio Edit tab.
    createLoraDownloadJob,
    // Global theme (sc-10244): the redesign top-bar ☾/☀ toggle drives the app-wide
    // data-theme, not a screen-local override — consistent with the rest of the app.
    theme = "light",
    changeTheme,
  } = useAppContext();
  // Under the keep-alive shell (sc-11959) the editor stays mounted (hidden) after the
  // user navigates away, so it can no longer rely on unmount to drop its leave-guard.
  // `true` unless a KeepAlivePane says this editor is backgrounded (defaults to true
  // for the direct-render unit tests, which carry no ScreenActiveContext).
  const screenActive = useScreenActive();
  // Mac UI gating (sc-3486): the upscale tool itself runs in-process on Rust (Real-ESRGAN,
  // sc-3489), so it is available on a gated Mac — this block is a defensive guard that stays
  // null. The second engine (AuraSR) is dropped on Mac (sc-3668) and gated per-engine below.
  const macUpscaleBlock = macFeatureBlock(macCapabilities, "imageUpscale");
  // Smart-select (sc-3751) runs native-MLX SAM3 — Mac-only, no torch/candle path. Gate it on the
  // platform-intrinsic `imageSegment` capability (true only on a Mac backend, false off-Mac and
  // pre-load), like the seedvr2 engine — independent of the Mac gating-rollout switch. When false,
  // the mask tool shows only the hand brush (graceful degradation).
  const smartSelectSupported = macCapabilities?.features?.imageSegment?.supported === true;

  // The working document (sc-6117): an ordered raster layer stack composited
  // bottom→top — `{ width, height, source, layers:[Layer], activeLayerId }` (see
  // ../imageLayers.js). A single-layer stack is the degenerate case that behaves
  // exactly like the pre-layers single bitmap, so the existing tools keep operating
  // on the active layer; the per-layer tool matrix + the panel land in sc-6118/6119.
  // Each live layer owns its decoded `image` + `objectUrl` (revoked on eviction).
  const [working, setWorking] = useState(null);
  const [status, setStatus] = useState({ loading: false, error: "" });
  const [pickerOpen, setPickerOpen] = useState(false);
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });

  // Redesign shell UI state (epic 10243). `layout` picks one of four panel
  // arrangements (accordion default) and persists to localStorage; `accCollapsed`
  // collapses the open accordion tool; `layersOpen` collapses the Layers block.
  const [layout, setLayoutState] = useState(readStoredEditorLayout);
  const setLayout = useCallback((next) => {
    setLayoutState(next);
    try {
      window.localStorage.setItem(EDITOR_LAYOUT_KEY, next);
    } catch {
      /* ignore (private mode etc.) */
    }
  }, []);
  const [accCollapsed, setAccCollapsed] = useState(false);
  const [layersOpen, setLayersOpen] = useState(true);
  // One undo checkpoint per opacity DRAG (mirrors LayersPanel): the first change of
  // a gesture snapshots; subsequent ticks coalesce until pointer-up resets it.
  const layerOpacityGestureRef = useRef(false);

  // Crop tool (sc-2430): client-side, rasterized into a new working image on Apply.
  const [tool, setTool] = useState("move");
  const [ratioKey, setRatioKey] = useState("free");
  const [rotated, setRotated] = useState(false);
  const [cropRect, setCropRect] = useState(null); // image-pixel coords, or null
  // Straighten (sc-10255): degrees the image is rotated before the axis-aligned crop
  // is rasterized on Apply (−15..15). 0 = no rotation (identical to the plain crop).
  const [straighten, setStraighten] = useState(0);
  // One undo checkpoint per Transform slider DRAG (mirrors the opacity gesture).
  const transformGestureRef = useRef(false);

  // Upscale tool (sc-2433): engine + factor for the in-flight request.
  const [upscaleEngine, setUpscaleEngine] = useState("real-esrgan");
  const [upscaleFactor, setUpscaleFactor] = useState(2);
  // SeedVR2 detail/softness knob (0..1, sc-4815) — only meaningful for the seedvr2 engine.
  const [upscaleSoftness, setUpscaleSoftness] = useState(0);
  // Engines offered in the picker; AuraSR is dropped on every platform (sc-3668 / sc-5499).
  const availableUpscaleEngines = upscaleEnginesForPlatform(macCapabilities);
  // If the selected engine got gated out (e.g. a stale saved AuraSR selection), fall back to the
  // default real-esrgan engine (the guaranteed-available cross-platform upscaler) so the tool stays
  // usable. Shared with ImageStudio via the single fallback hook (sc-8853).
  useUpscaleEngineFallback({
    macCapabilities,
    upscaleEngine,
    setUpscaleEngine,
    upscaleFactor,
    setUpscaleFactor,
  });

  // Per-tool logic lives in dedicated hooks now (sc-9752): the color-grade, mask, and
  // box tools each own their state/refs/handlers. They are called below (after the shared
  // callbacks they invoke — checkpoint / stagePointToImage / replaceLayerImage / runAiOp —
  // are defined) via stable ref bridges, so the hook call order stays fixed and each hook
  // always invokes the LATEST callback exactly as the pre-extraction inline closures did.
  const imageNodeRef = useRef(null); // Konva image node — cached for color-grade filtering + transform
  const histogramRef = useRef(null);
  // Stable bridges to callbacks defined later in this component. Assigned once those
  // callbacks exist; the tool hooks call `bridge.current(...)` so they read the latest
  // definition (identical to reading the live closure inline).
  const checkpointRef = useRef(() => {});
  const stagePointToImageRef = useRef(() => null);
  const replaceLayerImageRef = useRef(() => {});
  const runAiOpRef = useRef(() => {});
  const checkpointBridge = useCallback(() => checkpointRef.current(), []);
  const stagePointToImageBridge = useCallback((event) => stagePointToImageRef.current(event), []);
  const replaceLayerImageBridge = useCallback(
    (id, image, objectUrl, blob) => replaceLayerImageRef.current(id, image, objectUrl, blob),
    [],
  );
  const runAiOpBridge = useCallback((opts) => runAiOpRef.current(opts), []);

  // AI prompt edit (sc-2435): an edit-capable model + instruction + optional seed,
  // run against the working image through the existing edit_image flow.
  const editModels = editCapableModels(imageModels);
  const [editModel, setEditModel] = useState("");
  const [editPrompt, setEditPrompt] = useState("");
  const [editSeed, setEditSeed] = useState("");
  // Guidance override (sc-10275): "" = use the edit model's default (shown as the
  // input placeholder); a finite value rides advanced.guidanceScale.
  const [editGuidance, setEditGuidance] = useState("");
  // Canvas-extend / outpaint (sc-2556): target output aspect (default "match" = the
  // working size) and how to fill it (crop trims, pad bars, outpaint generates).
  const [editAspect, setEditAspect] = useState("match");
  const [editFitMode, setEditFitMode] = useState("crop");

  // Detail enhance (sc-2438): tile-ControlNet refine over the working image. Backbone
  // (SDXL/RealVisXL) + strength (the "detail amount" — higher invents more texture) +
  // structure-lock (controlnet scale). Defaults are the sc-2437 spike's locked recipe.
  const detailModels = detailCapableModels(imageModels);
  const [detailModel, setDetailModel] = useState("");
  const [detailStrength, setDetailStrength] = useState(0.55);
  const [detailCnScale, setDetailCnScale] = useState(0.7);
  // The tile ControlNet is a hard co-requisite of every detail run (worker `detail.rs`), but it ships as
  // a separate `type:"utility"` catalog artifact — so a detail-capable backbone can be installed while
  // the ControlNet is not, and the job would fail at run time. Surface it as a required dependency with a
  // one-click install, mirroring the managed edit-LoRA CTA below. Installed == not "missing" (App.jsx).
  const tileControlNet = tileControlNetModel(models);
  const tileControlNetReady = tileControlNetInstalled(models);
  const [tileControlNetDownloadRequested, setTileControlNetDownloadRequested] = useState(false);
  // Clear the transient "requested" state once the download lands (installState flips off "missing").
  useEffect(() => {
    if (tileControlNetReady) setTileControlNetDownloadRequested(false);
  }, [tileControlNetReady]);
  const requestTileControlNetDownload = useCallback(() => {
    if (!tileControlNet || !createModelDownloadJob) return;
    setTileControlNetDownloadRequested(true);
    createModelDownloadJob(tileControlNet);
  }, [tileControlNet, createModelDownloadJob]);

  // Keyboard-shortcut quick reference panel (sc-6111).
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  // Reference-image conditioning (sc-6107): user-attached library images that jointly condition the
  // AI Edit alongside the working image, on a FLUX.2 `multiReference` edit model. The working image is
  // added at run time (it's staged as a scratch source), so this holds only the user's picks.
  const [refAssetIds, setRefAssetIds] = useState([]); // string[] of library asset ids
  const [refPickerOpen, setRefPickerOpen] = useState(false);

  // Blank-canvas "New layout" (sc-6092): a from-scratch substrate for box layout
  // (Ideogram text-to-image). The modal picks an aspect + long-side size → W×H.
  const [newLayoutOpen, setNewLayoutOpen] = useState(false);
  const [layoutAspect, setLayoutAspect] = useState("1:1");
  const [layoutSize, setLayoutSize] = useState(1024);

  // Default the edit-model selection to the first edit-capable model once the model
  // list loads, and recover if the current pick stops being edit-capable.
  useEffect(() => {
    const caps = editCapableModels(imageModels);
    if (caps.length && !caps.some((model) => model.id === editModel)) setEditModel(caps[0].id);
  }, [imageModels, editModel]);

  // Same default/self-heal for the detail backbone.
  useEffect(() => {
    const caps = detailCapableModels(imageModels);
    if (caps.length && !caps.some((model) => model.id === detailModel)) setDetailModel(caps[0].id);
  }, [imageModels, detailModel]);

  // The chosen edit model + whether it accepts an inpaint mask (gates the mask tool).
  const selectedEditModel = editModels.find((model) => model.id === editModel) ?? null;
  const canMask = modelIsInpaintCapable(selectedEditModel);
  // Style/subject LoRAs for the AI Edit tool (sc-10254). Same family-gated selection +
  // serialization the studios use (useLoraSelection → serializeLora), threaded top-level
  // into buildEditJobBody; the worker's edit streams apply them via resolve_adapters.
  const editLoraSelection = useLoraSelection(loras, selectedEditModel);
  // ---- Krea-style managed image-edit LoRA (epic 10871, sc-11069) — parity with the Studio ----
  // The Krea 2 edit surface REQUIRES a dual-conditioning `image_edit` LoRA (worker R5) the base can't
  // edit without. Manage it for the user — auto-applied to the payload when installed (via
  // buildEditJobBody's `editLora`), surfaced as a one-click download when not — instead of leaving it
  // in the manual picker. `findModelEditLora` returns null for edit models that need none
  // (Qwen-Image-Edit, FLUX.2), so this whole block stays inert for them.
  const editLora = useMemo(() => findModelEditLora(loras, selectedEditModel), [loras, selectedEditModel]);
  const editLoraInstalled = loraIsInstalled(editLora);
  // The managed LoRA is applied automatically; hide it from the manual picker so it can't be
  // double-shown or accidentally toggled. Deduped again at payload time in case a stale selection
  // carries it (buildEditJobBody dedups by id).
  const managedEditLoraId = editLora && editLoraInstalled ? editLora.id : null;
  const editLoraRequiredMissing = Boolean(editLora) && !editLoraInstalled;
  const [editLoraDownloadRequested, setEditLoraDownloadRequested] = useState(false);
  // Clear the transient "requested" state once the download lands (installState flips) or the edit
  // LoRA leaves the picture (model change).
  useEffect(() => {
    if (!editLoraRequiredMissing) setEditLoraDownloadRequested(false);
  }, [editLoraRequiredMissing]);
  const requestEditLoraDownload = useCallback(() => {
    if (!editLora) return;
    setEditLoraDownloadRequested(true);
    createLoraDownloadJob?.(editLora);
  }, [editLora, createLoraDownloadJob]);
  // The manual LoRA picker hides the managed edit LoRA (it's applied automatically), so it can't be
  // double-shown or accidentally toggled — mirrors the Studio's pickerCompatibleLoras.
  const pickerCompatibleLoras = managedEditLoraId
    ? editLoraSelection.compatibleLoras.filter((lora) => lora.id !== managedEditLoraId)
    : editLoraSelection.compatibleLoras;
  // Whether the edit model conditions on extra reference images (FLUX.2 multi-reference edit, sc-6107):
  // the manifest tags it `ui.multiReference`. Gates the reference picker; off-models hide it entirely.
  const multiRefCapable = Boolean(selectedEditModel?.ui?.multiReference);
  // Drop any attached references when the model can't use them (switched away from a multiReference
  // model), so a stale selection never rides a job that would ignore it.
  useEffect(() => {
    if (!multiRefCapable && refAssetIds.length) setRefAssetIds([]);
  }, [multiRefCapable, refAssetIds.length]);

  // Save / export (sc-2434). `dirty` tracks edits not yet persisted to the Library;
  // `edits` is the ordered provenance chain; `savedAssetId` flags a completed Save
  // for the bar's "Saved" hint. A fresh open clears all three.
  const [dirty, setDirty] = useState(false);
  const [edits, setEdits] = useState([]);
  const [saving, setSaving] = useState(false);
  const [savedAssetId, setSavedAssetId] = useState(null);
  // An in-flight AI op (upscale now; AI-edit / detail later) on the working image.
  // The seam (sc-2432): stage the working bitmap as a scratch asset, run a worker
  // job against it, load the result back, then purge the scratch + result so the
  // session only persists on Save. { jobId, scratch (asset), source, label } | null.
  const [aiOp, setAiOp] = useState(null);

  const containerRef = useRef(null);
  const needsFitRef = useRef(false);
  // Monotonic layer-id source: ids survive an undo (the seq is snapshotted, like
  // boxIdSeq) so a layer added after an undo never collides with a recycled id.
  const layerIdRef = useRef(0);
  const cropRectRef = useRef(null);
  const transformerRef = useRef(null);
  const layerTransformerRef = useRef(null); // Konva transformer bound to the active layer (sc-6120)
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });

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
  // Live mirror of the working document for synchronous reads inside the commit
  // handlers (a checkpoint captures the pre-operation stack; restore reuses the
  // live decoded images for unchanged layers and revokes the URLs it drops).
  const workingRef = useRef(null);
  // Live mirrors of the snapshot-relevant state so a synchronous checkpoint can
  // capture the pre-operation state without stale-closure surprises.
  const editsRef = useRef(edits);
  const dirtyRef = useRef(dirty);
  const savedAssetIdRef = useRef(savedAssetId);
  const aiOpRef = useRef(aiOp);
  useEffect(() => { editsRef.current = edits; }, [edits]);
  useEffect(() => { dirtyRef.current = dirty; }, [dirty]);
  useEffect(() => { savedAssetIdRef.current = savedAssetId; }, [savedAssetId]);
  useEffect(() => { aiOpRef.current = aiOp; }, [aiOp]);
  useEffect(() => { workingRef.current = working; }, [working]);

  // ── Per-tool hooks (sc-9752, F-052 follow-up) ──────────────────────────────
  // Each tool owns its own state, refs, and handlers. They're called here — before the
  // snapshot/reset/pointer plumbing that reads their refs — and receive the shared,
  // late-defined callbacks through the stable ref bridges above (checkpoint /
  // stagePointToImage / replaceLayerImage / runAiOp), so the call order is fixed and each
  // hook invokes the LATEST callback exactly as the inline closures did. The boxes hook's
  // `boxesRef`/`boxColorRef`/`boxIdRef` are the SAME snapshot-mirror refs the editor reads
  // in captureSnapshot / writes in applyHistoryAux — the ref-mirror contract is preserved.
  const colorGradeTool = useColorGradeTool({
    working,
    tool,
    imageNodeRef,
    histogramRef,
    checkpoint: checkpointBridge,
    replaceLayerImage: replaceLayerImageBridge,
    blobToImage,
    setTool,
    setEdits,
    setDirty,
  });
  const {
    colorAdjust,
    colorMode,
    levels,
    curves,
    colorChannel,
    setColorMode,
    setColorChannel,
    setCurves,
    channelStroke,
    activeGradeIsIdentity,
    startColorGrade,
    setAdjustValue,
    resetAdjust,
    setLevelsValue,
    resetActiveColorMode,
    applyColorGrade,
    resetColorState,
    discardColorPreview,
  } = colorGradeTool;

  const boxesTool = useBoxesTool({
    working,
    tool,
    checkpoint: checkpointBridge,
    stagePointToImage: stagePointToImageBridge,
    setTool,
  });
  const {
    boxes,
    selectedBoxId,
    boxColor,
    boxDraft,
    setBoxes,
    setSelectedBoxId,
    setBoxColor,
    setBoxDraft,
    boxesRef,
    boxColorRef,
    boxIdRef,
    boxDrawingRef,
    boxTransformerRef,
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
  } = boxesTool;

  const maskTool = useMaskTool({
    working,
    tool,
    canMask,
    aiOp,
    activeProject,
    requestedGpu,
    runAiOp: runAiOpBridge,
    stagePointToImage: stagePointToImageBridge,
    blobToImage,
    setTool,
  });
  const {
    maskLines,
    maskMode,
    maskBrush,
    maskErase,
    maskRefineRadius,
    maskBaseImage,
    maskOverlay,
    maskSubTool,
    selectDraft,
    setMaskMode,
    setMaskBrush,
    setMaskErase,
    setMaskRefineRadius,
    setMaskSubTool,
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
  } = maskTool;

  // Memoize the image-renderable subset (sc-8939): the Image Editor re-renders on every
  // pointermove of a brush stroke / box drag, and re-filtering the full catalog each time
  // is needless work (jank on big projects). Only recompute when the catalog changes; this
  // also stabilizes the identity `imageAssets` feeds into the open-from-asset callback dep.
  const imageAssets = useMemo(() => (assets ?? []).filter(assetCanRenderAsImage), [assets]);

  // Track the container size so the Konva stage fills the available canvas area.
  // Measure once up front (a ResizeObserver alone can miss the first layout) and
  // then observe for later window / layout changes.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    const measure = () => setStageSize({ width: el.clientWidth, height: el.clientHeight });
    measure();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Revoke every live layer's object URL when the editor unmounts.
  useEffect(() => () => revokeLayerUrls(workingRef.current?.layers), []);

  const fitToView = useCallback(() => {
    if (!working || !stageSize.width || !stageSize.height) return;
    const scale = clamp(
      Math.min(stageSize.width / working.width, stageSize.height / working.height) * 0.92,
      MIN_SCALE,
      MAX_SCALE,
    );
    setView({
      scale,
      x: (stageSize.width - working.width * scale) / 2,
      y: (stageSize.height - working.height * scale) / 2,
    });
  }, [working, stageSize.width, stageSize.height]);

  // Fit a freshly loaded image once the stage has been measured (the stage may be
  // 0×0 on the first render before the ResizeObserver fires).
  useEffect(() => {
    if (needsFitRef.current && working && stageSize.width && stageSize.height) {
      needsFitRef.current = false;
      fitToView();
    }
  }, [working, stageSize.width, stageSize.height, fitToView]);

  const nextLayerId = () => `layer_${(layerIdRef.current += 1)}`;

  // Reset the per-bitmap editor overlays/tool state that a new working bitmap
  // invalidates (tool, crop, color preview, mask, references, boxes). Shared by
  // installWorkingImage (open/crop/AI result) and a bitmap-changing undo restore.
  const resetEditorOverlays = useCallback(() => {
    setTool("move");
    setCropRect(null);
    // Per-tool state resets are owned by each tool hook now (sc-9752). Each reset mirrors
    // the exact lines it replaced: color → adjust/levels/curves/mode/channel identity;
    // mask → strokes + smart-select base + sub-mode + select gesture latch; boxes →
    // boxes/selection/draft + node registry + draw latch.
    resetColorState();
    // A new working bitmap invalidates the mask (dims/content changed) — strokes + smart-select base.
    resetMaskState();
    // A new editing session starts with no attached reference images (sc-6107).
    setRefAssetIds([]);
    setRefPickerOpen(false);
    // Boxes are in image-pixel coords → a new bitmap (open/crop/upscale/AI op) invalidates them.
    resetBoxState();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Replace the whole working document with a fresh single-layer stack from one
  // decoded bitmap (open / blank / crop / color / AI result). Revokes the evicted
  // layers' object URLs first. The single-layer stack is the degenerate case that
  // reproduces the pre-layers single-bitmap behavior; multi-layer creation is the
  // layers panel (sc-6118).
  const installWorkingImage = useCallback(
    (image, objectUrl, blob, source) => {
      revokeLayerUrls(workingRef.current?.layers);
      needsFitRef.current = true;
      resetEditorOverlays();
      setWorking(singleLayerWorking({ id: nextLayerId(), image, objectUrl, blob, source }));
    },
    [resetEditorOverlays],
  );

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
  }, [captureSnapshot, syncHistoryFlags]);
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
    [resetEditorOverlays, applyHistoryAux],
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
  }, [captureSnapshot, restoreSnapshot, syncHistoryFlags]);

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
  }, [captureSnapshot, restoreSnapshot, syncHistoryFlags]);

  // ── Keyboard shortcuts (sc-6111) ───────────────────────────────────────────
  // One editor-scoped window keydown handler. Held behind a ref so the listener is
  // subscribed once (no add/remove churn during the high-frequency re-renders of a
  // crop / box / mask drag) while always seeing the latest tool + selection state.
  // Never fires while a text field is focused, so typing a prompt / box description
  // / renaming a layer is left to the browser. Undo/redo (sc-6106) are the only
  // modified combos we own; the rest are single keys that mirror the toolbar and the
  // zoom bar. `?` toggles the quick reference and works before an image is open.
  const onEditorKeyDownRef = useRef(null);
  onEditorKeyDownRef.current = (event) => {
    const tag = event.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || event.target?.isContentEditable) return;

    if (event.metaKey || event.ctrlKey) {
      const k = event.key?.toLowerCase();
      if (k === "z") {
        event.preventDefault();
        if (event.shiftKey) redo();
        else undo();
      } else if (k === "y") {
        event.preventDefault();
        redo();
      }
      return;
    }
    // Single-key shortcuts only — never with Alt (avoids hijacking OS combos).
    if (event.altKey) return;

    if (event.key === "?") {
      event.preventDefault();
      setShortcutsOpen((on) => !on);
      return;
    }
    if (event.key === "Escape") {
      if (shortcutsOpen) setShortcutsOpen(false);
      else escapeGesture();
      return;
    }

    if (!workingRef.current) return;

    // View shortcuts work regardless of the busy/AI state.
    switch (event.key) {
      case "+":
      case "=":
        event.preventDefault();
        zoomAtCenter(ZOOM_STEP);
        return;
      case "-":
      case "_":
        event.preventDefault();
        zoomAtCenter(1 / ZOOM_STEP);
        return;
      case "0":
        event.preventDefault();
        fitToView();
        return;
      case "1":
        event.preventDefault();
        actualSize();
        return;
      case "Delete":
      case "Backspace":
        if (tool === "boxes" && selectedBoxId) {
          event.preventDefault();
          deleteBox(selectedBoxId);
        }
        return;
      default:
        break;
    }

    // Tool switches. Move always works (it cancels/pans); the rest mirror their
    // toolbar buttons' enabled state and are suppressed while an AI op is running.
    const key = event.key.toLowerCase();
    if (key === "m") {
      cancelCrop();
      return;
    }
    if (aiOpRef.current) return;
    if (key === "t") startTransform();
    else if (key === "c") startCrop();
    else if (key === "u") {
      if (!macUpscaleBlock) setTool("upscale");
    } else if (key === "d") {
      if (detailModels.length) setTool("detail");
    } else if (key === "g") startColorGrade();
    else if (key === "e") setTool("edit");
    else if (key === "b") selectBoxTool();
  };

  useEffect(() => {
    const handler = (event) => onEditorKeyDownRef.current?.(event);
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const openFromBlob = useCallback(
    async (blob, source) => {
      setStatus({ loading: true, error: "" });
      try {
        const { image, objectUrl } = await blobToImage(blob);
        installWorkingImage(image, objectUrl, blob, source);
        // A freshly opened image is a clean session — clear edit/provenance state
        // and start a fresh undo/redo history rooted at this bitmap (sc-6106).
        setEdits([]);
        setDirty(false);
        setSavedAssetId(null);
        resetHistory();
        setStatus({ loading: false, error: "" });
      } catch (err) {
        setStatus({ loading: false, error: err.message || "Could not open image" });
      }
    },
    [installWorkingImage, resetHistory],
  );

  const openAsset = useCallback(
    async (assetId) => {
      const asset = imageAssets.find((item) => item.id === assetId);
      if (!asset) return;
      const url = assetUrl(asset);
      if (!url) {
        setStatus({ loading: false, error: "Asset has no media file" });
        return;
      }
      setStatus({ loading: true, error: "" });
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed to load asset (${res.status})`);
        const blob = await res.blob();
        await openFromBlob(blob, {
          kind: "asset",
          assetId: asset.id,
          name: asset.displayName ?? asset.id,
        });
      } catch (err) {
        setStatus({ loading: false, error: err.message || "Could not load asset" });
      }
    },
    [imageAssets, openFromBlob],
  );

  // sc-8730: consume the App-level Image Editor launch channel. When something outside
  // the editor (currently the FullscreenPreview "Edit" button via sendAssetToImageEditor)
  // routes an asset here, App switches activeView to "ImageEditor" and stashes
  // { id, assetId } in editorLaunch. Keyed on the launch id so it fires once per launch
  // (relaunching the same asset gets a fresh id) and never on unrelated re-renders.
  // Entering via the nav with no launch leaves editorLaunch null → no auto-open. We clear
  // the launch after consuming it so navigating away and back doesn't re-open a stale asset.
  useEffect(() => {
    if (!editorLaunch?.assetId) {
      return;
    }
    openAsset(editorLaunch.assetId);
    clearEditorLaunch?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editorLaunch?.id]);

  const openFile = useCallback(
    (file) => {
      if (!file || !file.type.startsWith("image/")) {
        setStatus({ loading: false, error: "Please choose an image file" });
        return;
      }
      openFromBlob(file, { kind: "upload", name: file.name });
    },
    [openFromBlob],
  );

  // Start a working-image session on a fresh blank (white) canvas (sc-6092). It
  // reuses the same session model as Open, then jumps into the box tool — the
  // point of a blank layout is to draw boxes and generate from them.
  const newBlankLayout = useCallback(
    async ({ width, height }) => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) {
        setStatus({ loading: false, error: "Could not create the canvas." });
        return;
      }
      await openFromBlob(blob, { kind: "blank", name: "Untitled layout" });
      setTool("boxes");
    },
    [openFromBlob],
  );

  async function createBlankLayout() {
    if (!(await confirmDiscardEdits())) return;
    setNewLayoutOpen(false);
    await newBlankLayout(blankCanvasDims(layoutAspect, layoutSize));
  }

  async function handleDrop(event) {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file && (await confirmDiscardEdits())) openFile(file);
  }

  function handleWheel(event) {
    event.evt.preventDefault();
    const stage = event.target.getStage();
    const pointer = stage?.getPointerPosition();
    if (!pointer) return;
    const oldScale = view.scale;
    const newScale = clamp(oldScale * (event.evt.deltaY > 0 ? 1 / ZOOM_STEP : ZOOM_STEP), MIN_SCALE, MAX_SCALE);
    const mouseTo = { x: (pointer.x - view.x) / oldScale, y: (pointer.y - view.y) / oldScale };
    setView({ scale: newScale, x: pointer.x - mouseTo.x * newScale, y: pointer.y - mouseTo.y * newScale });
  }

  function zoomAtCenter(factor) {
    const cx = stageSize.width / 2;
    const cy = stageSize.height / 2;
    const oldScale = view.scale;
    const newScale = clamp(oldScale * factor, MIN_SCALE, MAX_SCALE);
    const mouseTo = { x: (cx - view.x) / oldScale, y: (cy - view.y) / oldScale };
    setView({ scale: newScale, x: cx - mouseTo.x * newScale, y: cy - mouseTo.y * newScale });
  }

  function actualSize() {
    if (!working) return;
    setView({
      scale: 1,
      x: (stageSize.width - working.width) / 2,
      y: (stageSize.height - working.height) / 2,
    });
  }

  // ── Crop ────────────────────────────────────────────────────────────────
  function startCrop() {
    if (!working) return;
    setTool("crop");
    setStraighten(0);
    setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(ratioKey, rotated)));
  }

  function cancelCrop() {
    setTool("move");
    setCropRect(null);
    setStraighten(0);
    // Discard any unbaked color preview (adjust / levels / curves). Owned by the color
    // hook now (sc-9752), which resets exactly those three values and leaves mode/channel.
    discardColorPreview();
  }

  // Escape (sc-6111): cancel the most specific in-progress gesture, falling back to
  // deselecting / returning to the Move tool. Highest priority first.
  function escapeGesture() {
    if (boxDrawingRef.current) {
      boxDrawingRef.current = false;
      setBoxDraft(null);
      return;
    }
    // Cancel an in-flight smart-select drag (mask hook owns the gesture latch, sc-9752).
    if (cancelSelectDrag()) {
      return;
    }
    if (tool === "crop") {
      cancelCrop();
      return;
    }
    if (selectedBoxId) {
      setSelectedBoxId(null);
      return;
    }
    // Any other active tool → back to Move (also discards an unbaked color preview).
    if (tool !== "move") cancelCrop();
  }

  function chooseRatio(key) {
    setRatioKey(key);
    if (working) setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(key, rotated)));
  }

  function toggleRotate() {
    const next = !rotated;
    setRotated(next);
    if (working) setCropRect(centeredCropRect(working.width, working.height, cropRatioForKey(ratioKey, next)));
  }

  function clampCropToImage(rect) {
    const width = clamp(rect.width, MIN_CROP_PX, working.width);
    const height = clamp(rect.height, MIN_CROP_PX, working.height);
    return {
      width,
      height,
      x: clamp(rect.x, 0, working.width - width),
      y: clamp(rect.y, 0, working.height - height),
    };
  }

  function handleCropDragEnd() {
    const node = cropRectRef.current;
    if (!node) return;
    const next = clampCropToImage({ ...cropRect, x: node.x(), y: node.y() });
    node.position({ x: next.x, y: next.y });
    setCropRect(next);
  }

  function handleCropTransformEnd() {
    const node = cropRectRef.current;
    if (!node) return;
    const next = clampCropToImage({
      x: node.x(),
      y: node.y(),
      width: node.width() * node.scaleX(),
      height: node.height() * node.scaleY(),
    });
    node.scaleX(1);
    node.scaleY(1);
    node.setAttrs(next);
    setCropRect(next);
  }

  // ── Active-layer write-back + flatten plumbing (sc-6119) ───────────────────
  // Encode just the ACTIVE layer's bitmap to a PNG File — the source for an
  // active-layer AI op (same-size edit / detail / smart-select) whose result is
  // written back to that layer, the rest of the stack preserved.
  const activeLayerToFile = useCallback(
    (filename) =>
      new Promise((resolve, reject) => {
        const work = workingRef.current;
        const layer = activeLayerOf(work);
        if (!layer) {
          reject(new Error("No active layer."));
          return;
        }
        const canvas = document.createElement("canvas");
        canvas.width = layer.image.naturalWidth;
        canvas.height = layer.image.naturalHeight;
        canvas.getContext("2d").drawImage(layer.image, 0, 0);
        const base = (work.source.name || "image").replace(/\.[^./\\]+$/, "");
        canvas.toBlob(
          (blob) =>
            blob
              ? resolve(new File([blob], filename || `${base}.png`, { type: "image/png" }))
              : reject(new Error("Could not encode the layer.")),
          "image/png",
        );
      }),
    [],
  );

  // Write a decoded AI/grade result back into a specific layer, revoking that
  // layer's previous object URL. Preserves the doc dims + the rest of the stack.
  const replaceLayerImage = useCallback((id, image, objectUrl, blob) => {
    const prev = layerById(workingRef.current, id);
    if (prev?.objectUrl && prev.objectUrl !== objectUrl) URL.revokeObjectURL(prev.objectUrl);
    setWorking((cur) => replaceLayerBitmap(cur, id, { image, objectUrl, blob }));
  }, []);
  // Keep the color-grade hook's bridge pointed at the latest replaceLayerImage (sc-9752).
  replaceLayerImageRef.current = replaceLayerImage;

  // A document-level AI op (upscale / outpaint / box-keyed edit) flattens the stack
  // into one base layer; warn before discarding a multi-layer stack.
  const confirmFlatten = useCallback(() => {
    const n = workingRef.current?.layers?.length ?? 0;
    if (n <= 1) return true;
    // Desktop-safe confirm (sc-11968): returns a Promise the caller awaits.
    return appConfirm({
      title: "Flatten layers?",
      message: `This will flatten ${n} layers into a single layer. Continue?`,
      confirmLabel: "Flatten",
      cancelLabel: "Cancel",
    });
  }, []);

  // Apply: document-level crop — crop every layer to the rect, set the doc dims,
  // keep the stack. The bitmaps are blob-backed (never tainted), so reading pixels
  // back is safe; provenance is preserved so lineage survives to Save (sc-2434).
  const applyCrop = useCallback(async () => {
    if (!working || !cropRect || !working.layers.length) return;
    const sx = clamp(Math.round(cropRect.x), 0, working.width - 1);
    const sy = clamp(Math.round(cropRect.y), 0, working.height - 1);
    const sw = clamp(Math.round(cropRect.width), 1, working.width - sx);
    const sh = clamp(Math.round(cropRect.height), 1, working.height - sy);
    // Document-level crop (sc-6119): crop EVERY layer's bitmap to the rect and set the
    // doc dims — the stack is preserved. Layers are doc-aligned in this slice; per-layer
    // transforms (and transform-aware crop) arrive with sc-6120.
    let cropped;
    try {
      cropped = await Promise.all(
        working.layers.map(async (layer) => {
          const canvas = document.createElement("canvas");
          canvas.width = sw;
          canvas.height = sh;
          const ctx = canvas.getContext("2d");
          if (straighten) {
            // Straighten (sc-10255): rotate the layer by `straighten°` about the crop-rect
            // centre, then take the axis-aligned sw×sh window — a rotate-then-crop. Corners
            // that rotate past the source edge come through transparent (inset your crop).
            const cx = sx + sw / 2;
            const cy = sy + sh / 2;
            ctx.translate(sw / 2, sh / 2);
            ctx.rotate((straighten * Math.PI) / 180);
            ctx.drawImage(layer.image, -cx, -cy);
          } else {
            ctx.drawImage(layer.image, sx, sy, sw, sh, 0, 0, sw, sh);
          }
          const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
          if (!blob) throw new Error("Could not encode the crop.");
          const { image, objectUrl } = await blobToImage(blob);
          return { id: layer.id, image, objectUrl, blob };
        }),
      );
    } catch (err) {
      setStatus({ loading: false, error: err.message || "Could not crop the layers." });
      return;
    }
    checkpoint();
    const oldLayers = workingRef.current.layers;
    needsFitRef.current = true;
    // Crop invalidates the mask + boxes (old-document pixel coords) and returns to Move.
    resetEditorOverlays();
    const byId = new Map(cropped.map((c) => [c.id, c]));
    setWorking((prev) => ({
      ...prev,
      width: sw,
      height: sh,
      layers: prev.layers.map((layer) => {
        const c = byId.get(layer.id);
        return c ? { ...layer, image: c.image, objectUrl: c.objectUrl, blob: c.blob } : layer;
      }),
    }));
    oldLayers.forEach((layer) => layer.objectUrl && URL.revokeObjectURL(layer.objectUrl));
    setEdits((prev) => [...prev, { op: "crop", width: sw, height: sh, ...(straighten ? { straighten } : {}) }]);
    setStraighten(0);
    setDirty(true);
  }, [working, cropRect, straighten, checkpoint, resetEditorOverlays]);

  // Bind the transformer to the crop rect whenever crop mode is active.
  useEffect(() => {
    const transformer = transformerRef.current;
    const node = cropRectRef.current;
    if (tool === "crop" && transformer && node) {
      transformer.nodes([node]);
      transformer.getLayer()?.batchDraw();
    }
  }, [tool, cropRect]);

  // Bind the layer transformer to the ACTIVE layer's node whenever the Transform
  // tool is active (sc-6120); re-bind when the active layer changes. `working` in
  // the deps covers an active-layer switch (imageNodeRef reattaches to the new node).
  useEffect(() => {
    const transformer = layerTransformerRef.current;
    if (!transformer) return;
    const node = tool === "transform" ? imageNodeRef.current : null;
    transformer.nodes(node ? [node] : []);
    transformer.getLayer()?.batchDraw();
  }, [tool, working]);

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
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
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

  // Flatten the visible layer stack onto a fresh canvas at the document size
  // (sc-6117). The layers' images are already decoded, so this is synchronous;
  // callers toBlob it (Save / Download / AI-op source) or paint overlays on top
  // first (the box-keyed edit). The shared composite behind every editor export.
  function compositeToCanvas(work = working) {
    const canvas = document.createElement("canvas");
    canvas.width = work.width;
    canvas.height = work.height;
    compositeLayersToCanvas(canvas.getContext("2d"), work.layers, { visibleOnly: true });
    return canvas;
  }

  // Rasterize the composited document + the colored boxes into one PNG File (sc-6093).
  // This is an ephemeral pass-through reference — staged as scratch, never saved
  // to the Library — that the edit model reads as color-keyed regions.
  function bakeBoxesToFile() {
    return new Promise((resolve, reject) => {
      const canvas = compositeToCanvas();
      paintBoxesOnContext(canvas.getContext("2d"), boxes);
      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error("Could not bake the boxes."));
          return;
        }
        resolve(new File([blob], "boxed.png", { type: "image/png" }));
      }, "image/png");
    });
  }

  // Bake the boxes and run them through the existing edit_image flow on the chosen
  // edit model (sc-6093). The baked PNG is the pass-through source; runAiOp stages
  // it as scratch and purges it with the result, so it never lands in the Library.
  async function runBoxEdit() {
    if (!boxes.length || !editModel || !working || aiOp) return;
    const prompt = editPrompt.trim();
    let sourceFile;
    try {
      sourceFile = await bakeBoxesToFile();
    } catch (err) {
      setStatus({ loading: false, error: `Could not bake boxes: ${err.message || err}` });
      return;
    }
    runAiOp({
      label: "edit",
      endpoint: "/api/v1/image/jobs",
      // The boxes overlay the whole document → the baked composite is the source and
      // the re-rendered result flattens the stack to one base layer (sc-6119).
      layerSource: "composite",
      edit: { op: "boxLayout", model: editModel, prompt, boxes: boxes.length },
      sourceFile,
      buildBody: (scratch) =>
        buildEditJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          model: editModel,
          prompt,
          seed: editSeed,
          width: working.width,
          height: working.height,
          fitMode: "crop",
          // The boxes-layout edit runs the same edit model, so it also needs the managed
          // image-edit LoRA when the model requires one (Krea R5) — sc-11069.
          editLora: managedEditLoraId ? editLora : null,
          // Identity strength (sc-11798): the user's edit-LoRA weight override, or the default.
          editLoraWeight: managedEditLoraId ? editLoraSelection.weightFor(editLora) : null,
        }),
    });
  }

  // The stage's pointer events drive both the mask brush (edit tool) and box
  // drawing (boxes tool); each handler no-ops unless its tool/mode is active.
  function handleStagePointerDown(event) {
    maskPointerDown(event);
    selectPointerDown(event);
    boxPointerDown(event);
  }
  function handleStagePointerMove(event) {
    maskPointerMove(event);
    selectPointerMove(event);
    boxPointerMove(event);
  }
  function handleStagePointerUp(event) {
    maskPointerUp(event);
    selectPointerUp(event);
    boxPointerUp(event);
  }

  // ── Inpaint mask brush (sc-2436) ──────────────────────────────────────────
  // Pointer position in image-pixel coords (undo the stage pan/zoom), clamped. Stays in
  // the editor (it reads `view` + `working` + the shared `clamp`) and is bridged into the
  // mask + boxes hooks via stagePointToImageRef so they read the latest closure (sc-9752).
  function stagePointToImage(event) {
    const stage = event.target.getStage();
    const pointer = stage?.getPointerPosition();
    if (!pointer || !working) return null;
    return {
      x: clamp((pointer.x - view.x) / view.scale, 0, working.width),
      y: clamp((pointer.y - view.y) / view.scale, 0, working.height),
    };
  }
  stagePointToImageRef.current = stagePointToImage;

  // ── AI ops on the working image (sc-2432 seam) ────────────────────────────
  // Flatten the composited document to a PNG File. `filename` overrides the name
  // (Save/Download use the "-edited" name; the AI-op scratch upload doesn't care).
  const workingImageToFile = useCallback(
    (filename) => {
      return new Promise((resolve, reject) => {
        if (!working) {
          reject(new Error("No working image."));
          return;
        }
        const canvas = compositeToCanvas(working);
        const base = (working.source.name || "image").replace(/\.[^./\\]+$/, "");
        const name = filename || `${base}.png`;
        canvas.toBlob((blob) => {
          if (!blob) {
            reject(new Error("Could not encode the working image."));
            return;
          }
          resolve(new File([blob], name, { type: "image/png" }));
        }, "image/png");
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [working],
  );

  // Stage the working image as a scratch asset, start a worker job against it, and
  // track it. The watcher below loads the result back and purges scratch + result —
  // intermediates never persist; only Save (sc-2434) lands a Library asset.
  const runAiOp = useCallback(
    async ({
      buildBody,
      label,
      edit,
      endpoint = "/api/v1/jobs",
      maskFile = null,
      sourceFile = null,
      onComplete = null,
      // The tool-interaction matrix (sc-6119): "activeLayer" ops stage the active
      // layer and write the result back to it (dims unchanged); "composite" ops
      // stage the flattened document and the result becomes a single new base layer.
      layerSource = "activeLayer",
    }) => {
      if (!working || aiOp || !activeProject) return;
      // A composite-source op flattens the stack into one base layer — confirm first.
      if (layerSource === "composite" && !(await confirmFlatten())) return;
      setStatus({ loading: false, error: "" });
      const targetLayerId = activeLayerOf(working)?.id ?? null;
      // Stage the source (and, for a masked edit, the mask) as scratch assets. An
      // explicit sourceFile (e.g. the box-baked pass-through, sc-6093) wins; else
      // stage the composite (flatten ops) or just the active layer (active-layer ops).
      let scratch;
      let maskScratch = null;
      try {
        const staged =
          sourceFile ?? (layerSource === "composite" ? await workingImageToFile() : await activeLayerToFile());
        scratch = await importAsset(staged, { throwOnError: true });
        if (maskFile) maskScratch = await importAsset(maskFile, { throwOnError: true });
      } catch (err) {
        if (scratch) purgeAsset(scratch).catch(() => {});
        setStatus({ loading: false, error: `Could not stage image: ${err.message || err}` });
        return;
      }
      try {
        const job = await apiFetch(endpoint, token, {
          method: "POST",
          body: JSON.stringify(buildBody(scratch, maskScratch)),
        });
        if (!job?.id) throw new Error("The job was not created.");
        // Register the scratch op with App so its scratch/mask (and later result) assets
        // are purged even if the user navigates away mid-job and this editor unmounts
        // before the completion watcher below runs (sc-8850).
        trackEditorScratchOp?.(job.id, [scratch, maskScratch]);
        setAiOp({
          jobId: job.id,
          scratch,
          maskScratch,
          source: working.source,
          label,
          edit,
          onComplete,
          // How the watcher writes the result back: active-layer ops replace the
          // target layer's bitmap; composite ops flatten the stack to one layer.
          writeBack: onComplete ? null : layerSource === "composite" ? "document" : "activeLayer",
          targetLayerId,
        });
        setTool("move");
      } catch (err) {
        purgeAsset(scratch).catch(() => {});
        if (maskScratch) purgeAsset(maskScratch).catch(() => {});
        setStatus({ loading: false, error: `Could not start ${label}: ${err.message || err}` });
      }
    },
    [working, aiOp, activeProject, workingImageToFile, activeLayerToFile, confirmFlatten, importAsset, token, purgeAsset, trackEditorScratchOp],
  );
  // Keep the mask hook's bridge pointed at the latest runAiOp (sc-9752) — runSmartSelect
  // stages a scratch image_segment job through this exact seam.
  runAiOpRef.current = runAiOp;

  function runUpscale() {
    const valid = upscaleFactorsForEngine(upscaleEngine);
    const factor = valid.includes(upscaleFactor) ? upscaleFactor : valid[0];
    const softness = upscaleEngineHasSoftness(upscaleEngine) ? upscaleSoftness : undefined;
    runAiOp({
      label: "upscale",
      // Upscale changes dimensions → document-level: flatten the stack, upscale once.
      layerSource: "composite",
      edit: {
        op: "upscale",
        engine: upscaleEngine,
        factor,
        ...(softness !== undefined ? { softness } : {}),
      },
      buildBody: (scratch) =>
        buildUpscaleJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          factor,
          engine: upscaleEngine,
          displayName: working?.source?.name,
          softness,
        }),
    });
  }

  function runDetail() {
    if (!detailModel) return;
    runAiOp({
      label: "detail",
      edit: { op: "detail", model: detailModel, strength: detailStrength, cnScale: detailCnScale },
      buildBody: (scratch) =>
        buildDetailJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          model: detailModel,
          strength: detailStrength,
          cnScale: detailCnScale,
          displayName: working?.source?.name,
        }),
    });
  }

  async function runEdit() {
    const prompt = editPrompt.trim();
    if (!prompt || !editModel || !working) return;
    // A required image-edit LoRA that isn't downloaded yet blocks the run (worker R5): the source
    // band renders the actionable Download note (epic 10871, sc-11069). Defensive — the Generate
    // button is already disabled on this condition.
    if (editLoraRequiredMissing) return;
    // Canvas-extend / outpaint (sc-2556): resolve the output W×H from the chosen aspect
    // and fit mode (outpaint coerced away when the model can't inpaint). "match" keeps
    // the working size, so the existing same-size edit behavior is unchanged.
    const fitMode = effectiveFitMode(editFitMode, canMask);
    const { width: outWidth, height: outHeight } = editOutputDims(working.width, working.height, editAspect, fitMode);
    // Same-size edit → active layer; a canvas-extend / outpaint (dims change) →
    // document-level flatten (sc-6119 tool matrix).
    const dimsChange = outWidth !== working.width || outHeight !== working.height;
    // A mask is sent only for inpaint-capable models; otherwise it's a whole-image edit (the mask
    // stays as a local guide but isn't uploaded). The mask is brush strokes (sc-2436) and/or a
    // smart-select base (sc-3751), composited together by rasterizeMaskToFile.
    const masked = canMask && (maskHasContent(maskLines) || Boolean(maskBaseImage));
    let maskFile = null;
    if (masked) {
      try {
        maskFile = await rasterizeMaskToFile();
      } catch (err) {
        setStatus({ loading: false, error: `Could not prepare the mask: ${err.message || err}` });
        return;
      }
    }
    runAiOp({
      label: "edit",
      endpoint: "/api/v1/image/jobs",
      layerSource: dimsChange ? "composite" : "activeLayer",
      edit: { op: "edit", model: editModel, prompt, ...(masked ? { masked: true } : {}) },
      maskFile,
      buildBody: (scratch, maskScratch) =>
        buildEditJobBody({
          project: activeProject,
          requestedGpu,
          sourceAssetId: scratch.id,
          maskAssetId: maskScratch?.id,
          // Multi-reference edit (sc-6107): lead with the working scratch image, then the user's
          // references. Only for a multiReference model with at least one attached reference.
          referenceAssetIds:
            multiRefCapable && refAssetIds.length ? editReferenceIds(scratch.id, refAssetIds) : null,
          model: editModel,
          prompt,
          seed: editSeed,
          width: outWidth,
          height: outHeight,
          fitMode,
          loras: editLoraSelection.serializedLoras,
          // Auto-apply the model's managed image-edit LoRA (R5) when installed — deduped inside
          // buildEditJobBody, so a run needs no manual picking (epic 10871, sc-11069).
          editLora: managedEditLoraId ? editLora : null,
          // Identity strength (sc-11798): the user's edit-LoRA weight override, or the default.
          editLoraWeight: managedEditLoraId ? editLoraSelection.weightFor(editLora) : null,
          guidanceScale: editGuidance,
        }),
    });
  }

  // When the in-flight op's job terminates, load the result back into the working
  // image (on success) and purge the ephemeral scratch + result assets.
  useEffect(() => {
    if (!aiOp?.jobId) return;
    const job = jobs?.find((item) => item.id === aiOp.jobId);
    if (!job || !terminalStatuses.has(job.status)) return;
    const { jobId, source, edit, onComplete, writeBack, targetLayerId } = aiOp;
    setAiOp(null); // stop tracking immediately so this can't re-enter on the next jobs tick
    const resultAsset = job.status === "completed" ? job.result?.assets?.[0] ?? null : null;
    (async () => {
      try {
        if (!resultAsset) {
          setStatus({ loading: false, error: job.error ?? job.message ?? "The operation failed." });
          return;
        }
        // Smart-select (sc-3751): the caller's `onComplete` consumes the result asset itself (loads
        // the mask into the mask layer) — it does NOT replace the working image, so skip the install
        // / history / dirty path entirely.
        if (onComplete) {
          await onComplete(resultAsset);
          return;
        }
        const res = await fetch(assetUrl(resultAsset));
        if (!res.ok) throw new Error(`Failed to load result (${res.status})`);
        const blob = await res.blob();
        const { image, objectUrl } = await blobToImage(blob);
        checkpoint();
        // Active-layer op (same-size edit / detail) → write the result back into the
        // target layer, preserving the rest of the stack; document op (upscale /
        // outpaint / box edit) → flatten the stack into one new base layer (sc-6119).
        if (writeBack === "activeLayer" && targetLayerId && layerById(workingRef.current, targetLayerId)) {
          replaceLayerImage(targetLayerId, image, objectUrl, blob);
        } else {
          installWorkingImage(image, objectUrl, blob, source);
        }
        if (edit) setEdits((prev) => [...prev, edit]);
        setDirty(true);
      } catch (err) {
        setStatus({ loading: false, error: err.message || "The operation failed." });
      } finally {
        // Hand the purge to App (sc-8850): it owns the scratch registry, so it purges the
        // scratch + mask + result assets through the single survivor path. This also drops
        // the registry entry so the App-level sweep won't double-purge. The result is
        // loaded into the canvas above BEFORE this releases it — intermediates never persist.
        releaseEditorScratchOp?.(jobId, job);
      }
    })();
  }, [aiOp, jobs, installWorkingImage, replaceLayerImage, releaseEditorScratchOp, checkpoint]);

  // ── Save / export (sc-2434) ───────────────────────────────────────────────
  // Persist the working image as a NEW Library asset, never overwriting the
  // source. Lineage links it back to the asset it was opened from (uploads have
  // no source to link); the edit chain rides along as provenance.
  const runSave = useCallback(async () => {
    if (!working || saving) return;
    setSaving(true);
    setStatus({ loading: false, error: "" });
    try {
      const file = await workingImageToFile(editedFilename(working.source));
      const saved = await importAsset(file, {
        throwOnError: true,
        sourceAssetId: working.source.assetId,
        provenance: buildSaveProvenance({
          source: working.source,
          edits,
          width: working.width,
          height: working.height,
          layers: working.layers,
        }),
      });
      setSavedAssetId(saved?.id ?? null);
      setDirty(false);
    } catch (err) {
      setStatus({ loading: false, error: `Could not save: ${err.message || err}` });
    } finally {
      setSaving(false);
    }
  }, [working, saving, workingImageToFile, importAsset, edits]);

  // Export the working image straight to disk as a PNG (no project involvement).
  const runDownload = useCallback(async () => {
    if (!working) return;
    try {
      const file = await workingImageToFile(editedFilename(working.source));
      const url = URL.createObjectURL(file);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = file.name;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setStatus({ loading: false, error: `Could not export: ${err.message || err}` });
    }
  }, [working, workingImageToFile]);

  // Confirm before an action that would discard unsaved edits (Open / drag-drop a
  // new image while dirty). Resolves true when it's safe to proceed. Async + desktop-safe
  // (sc-11968): callers `await` it, so the confirm works in the Tauri WebView where a raw
  // window.confirm silently no-ops.
  function confirmDiscardEdits() {
    if (!dirty) return Promise.resolve(true);
    return appConfirm({
      title: "Discard unsaved edits?",
      message: "You have unsaved edits. Open a new image and discard them?",
      confirmLabel: "Discard & open",
      cancelLabel: "Keep editing",
      tone: "danger",
    });
  }

  // Explicitly close the working document (the top-bar Close button, sc-11968). Under
  // keep-alive the editor no longer unmounts on navigation, so this is the intentional
  // path that clears the working doc, edit/undo history, and save state. A dirty doc (or
  // an in-flight AI op) prompts first via the desktop-safe confirm; a clean doc closes
  // silently. Clearing `aiOp` drops the survivor claim, so App's scratch registry
  // (editorScratch.js) purges any in-flight op's scratch/result when its job terminates —
  // nothing is orphaned.
  const closeDoc = useCallback(async () => {
    if (!workingRef.current) return;
    const message = closeConfirmMessage({ dirty: dirtyRef.current, aiOpPending: Boolean(aiOpRef.current) });
    if (message) {
      const proceed = await appConfirm({
        title: "Close image?",
        message,
        confirmLabel: "Discard & close",
        cancelLabel: "Keep editing",
        tone: "danger",
      });
      if (!proceed) return;
    }
    revokeLayerUrls(workingRef.current?.layers);
    setWorking(null);
    setEdits([]);
    setDirty(false);
    setSavedAssetId(null);
    setAiOp(null);
    resetHistory();
    setStatus({ loading: false, error: "" });
  }, [resetHistory]);

  // Warn before leaving with unsaved edits OR an in-flight AI op (sc-2434 / sc-8850): a
  // browser unload (close/refresh) and an in-app navigation away (the App nav consults
  // this guard). Starting an AI op does NOT set `dirty` — its result only lands on
  // success — so the guard must also fire while `aiOp` is non-null, otherwise the user
  // can silently navigate away and abandon the running op (losing the result and, before
  // the App-level survivor sweep, orphaning the scratch upload).
  const aiOpPending = Boolean(aiOp);
  useEffect(() => {
    // See leaveGuardArming: the beforeunload handler arms whenever there are unsaved
    // edits / an in-flight op (even backgrounded); the in-app nav guard only arms while
    // this editor is foregrounded (sc-11959).
    const { message, beforeUnload, inApp } = leaveGuardArming({ dirty, aiOpPending, screenActive });
    if (!beforeUnload) return undefined;
    const onBeforeUnload = (event) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    // Desktop-safe in-app leave guard (sc-11968): the guard returns a Promise<boolean> from
    // appConfirm (a real dialog), which navTo awaits before switching views — replacing the
    // raw window.confirm that silently no-ops in the Tauri WebView.
    const unregister = inApp
      ? registerLeaveGuard?.(() =>
          appConfirm({
            title: "Leave the Image Editor?",
            message,
            confirmLabel: "Leave",
            cancelLabel: "Stay",
            tone: "danger",
          }),
        )
      : undefined;
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      if (typeof unregister === "function") unregister();
    };
  }, [dirty, aiOpPending, registerLeaveGuard, screenActive]);

  // Claim the in-flight AI op's jobId with App so its survivor sweep (sc-8850) knows this
  // editor is alive and owns loading the result back before the scratch/result assets are
  // purged. The getter reads live `aiOp` via the ref, so the claim registration itself is
  // stable — it only unregisters (and triggers App's post-unmount sweep) when this editor
  // unmounts. An op that completes after this unmount is then purged by App, not lost here.
  useEffect(() => {
    if (!registerEditorScratchClaim) return undefined;
    return registerEditorScratchClaim(() => {
      const id = aiOpRef.current?.jobId;
      return id ? new Set([id]) : new Set();
    });
  }, [registerEditorScratchClaim]);

  const activeAiJob = aiOp ? jobs?.find((item) => item.id === aiOp.jobId) : null;

  // The box currently selected for metadata editing (sc-6091), and what it still
  // needs to be a valid Ideogram element (surfaced as a hint, not a hard block).
  const selectedBox = selectedBoxId ? boxes.find((box) => box.id === selectedBoxId) ?? null : null;
  const selectedBoxGaps = boxMetadataGaps(selectedBox);

  // Live W×H preview for the New-layout modal (sc-6092).
  const layoutDims = blankCanvasDims(layoutAspect, layoutSize);

  // The auto-composed color-keyed prompt from the current boxes (sc-6094). Used to
  // pre-fill the prompt field on demand; "" when no box is describable yet.
  const composedPrompt = composeColorPrompt(boxes);

  // ── Redesign shell derived values / dispatch (epic 10243) ──────────────────
  const activeMeta = EDITOR_TOOL_META[tool];
  const zoomPct = Math.round(view.scale * 100);
  const layerCount = working ? working.layers.length : 0;
  const docName = working ? working.source.name : "No image open";
  const docFormat = working ? (working.source.name?.split(".").pop() || "png").toUpperCase() : "";
  const docSub = working ? `${working.width} × ${working.height} · ${docFormat}` : "No document";
  // Accordion open only when the tool's panel isn't collapsed (accordion mode only).
  const panelOpen = !(layout === "accordion" && accCollapsed);
  const maskActive = canMask && (maskHasContent(maskLines) || Boolean(maskBaseImage));

  function toolIsDisabled(key) {
    if (key === "move") return false;
    if (aiOp) return true;
    if (key === "upscale") return Boolean(macUpscaleBlock);
    if (key === "detail") return detailModels.length === 0;
    return false;
  }
  // Route each tool through its existing entry handler (some prime state, e.g. crop
  // rect / transform target / color preview) — mirrors the pre-redesign toolbar.
  function selectTool(key) {
    if (toolIsDisabled(key)) return;
    if (key === "move") cancelCrop();
    else if (key === "transform") startTransform();
    else if (key === "crop") startCrop();
    else if (key === "color") startColorGrade();
    else if (key === "boxes") selectBoxTool();
    else setTool(key);
  }
  // Accordion header: clicking the open tool collapses it; any other selects + expands.
  function onAccordionHead(key) {
    if (tool === key) setAccCollapsed((v) => !v);
    else {
      setAccCollapsed(false);
      selectTool(key);
    }
  }
  const toolHint = {
    move: "Drag to pan · scroll to zoom",
    transform: "Drag the handles on the canvas to move, scale or rotate",
    crop: "Drag the crop handles, or set an exact size on the right",
    upscale: "Pick an engine and factor, then run",
    detail: "Tune detail & structure, then enhance",
    color: "Grade with adjust, levels or curves",
    edit: maskMode ? "Paint or box-select the region to edit" : "Describe the edit on the right",
    boxes: "Drag to draw a region, then describe it",
  }[tool];

  // Short engine blurbs for the upscale radio cards (design copy). Keyed by the
  // platform engine list (`availableUpscaleEngines`); unknown keys get no blurb.
  const UPSCALE_ENGINE_DESC = {
    "real-esrgan": "Fast, faithful general-purpose upscaler. Great default.",
    seedvr2: "Detail-restoring diffusion upscaler for degraded sources.",
    "aura-sr": "Sharpest output, best for print. Slower.",
  };
  const setCropDim = (dim, raw) => {
    const value = Number(raw);
    if (!cropRect || !working || !Number.isFinite(value)) return;
    setCropRect(clampCropToImage({ ...cropRect, [dim]: value }));
  };

  const renderToolPanel = (key) => {
    switch (key) {
      case "move":
        return (
          <>
            <div className="ie-section">
              <div className="ie-sec-title">Document</div>
              <div className="ie-readout">
                <span className="ie-readout-k">Dimensions</span>
                <span className="ie-readout-v">
                  {working.width} × {working.height}
                </span>
              </div>
              <div className="ie-readout">
                <span className="ie-readout-k">Layers</span>
                <span className="ie-readout-v">{layerCount}</span>
              </div>
              <p className="ie-note">
                Drag on the canvas to pan. Scroll to zoom. Pick a tool to start editing — each tool&apos;s controls appear
                here.
              </p>
            </div>
            <div className="ie-section">
              <div className="ie-sec-title">Quick actions</div>
              <button className="ie-btn block" onClick={fitToView} type="button">
                Fit to view
              </button>
              <button className="ie-btn block" onClick={actualSize} type="button">
                Actual size (100%)
              </button>
            </div>
          </>
        );
      case "transform": {
        const tLayer = activeLayerOf(working);
        const t = tLayer?.transform ?? identityTransform();
        const scalePct = Math.round(Math.abs(t.scaleX) * 100);
        const signX = t.scaleX < 0 ? -1 : 1;
        const signY = t.scaleY < 0 ? -1 : 1;
        return (
          <>
            <div className="ie-section">
              <div className="ie-sec-title">Position</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">X</span>
                  </div>
                  <input
                    className="ie-input ie-numfield"
                    onChange={(event) => setActiveTransform({ x: Number(event.target.value) || 0 })}
                    type="number"
                    value={Math.round(t.x)}
                  />
                </div>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Y</span>
                  </div>
                  <input
                    className="ie-input ie-numfield"
                    onChange={(event) => setActiveTransform({ y: Number(event.target.value) || 0 })}
                    type="number"
                    value={Math.round(t.y)}
                  />
                </div>
              </div>
            </div>
            <div className="ie-section">
              <div className="ie-sec-title">Scale &amp; rotation</div>
              <div className="ie-field">
                <div className="ie-field-top">
                  <span className="ie-field-label">Scale</span>
                  <span className="ie-field-val">{scalePct}%</span>
                </div>
                <input
                  className="ie-range"
                  max={300}
                  min={10}
                  onBlur={endTransformGesture}
                  onChange={(event) => {
                    const pct = Number(event.target.value) / 100;
                    onTransformSlider({ scaleX: signX * pct, scaleY: signY * pct });
                  }}
                  onMouseUp={endTransformGesture}
                  onTouchEnd={endTransformGesture}
                  type="range"
                  value={scalePct}
                />
              </div>
              <div className="ie-field">
                <div className="ie-field-top">
                  <span className="ie-field-label">Rotation</span>
                  <span className="ie-field-val">{Math.round(t.rotation)}°</span>
                </div>
                <input
                  className="ie-range"
                  max={180}
                  min={-180}
                  onBlur={endTransformGesture}
                  onChange={(event) => onTransformSlider({ rotation: Number(event.target.value) })}
                  onMouseUp={endTransformGesture}
                  onTouchEnd={endTransformGesture}
                  type="range"
                  value={Math.round(t.rotation)}
                />
              </div>
              <div className="ie-seg two" style={{ width: "100%" }}>
                <button className="ie-seg-btn" onClick={() => flipActiveLayer("h")} type="button">
                  Flip horizontal
                </button>
                <button className="ie-seg-btn" onClick={() => flipActiveLayer("v")} type="button">
                  Flip vertical
                </button>
              </div>
            </div>
            <div className="ie-section">
              <p className="ie-note">You can also drag the handles on the canvas to move, scale or rotate the layer.</p>
              <button className="ie-btn block" onClick={resetActiveLayerTransform} type="button">
                Reset transform
              </button>
              <button className="ie-btn block primary" onClick={() => setTool("move")} type="button">
                Done
              </button>
            </div>
          </>
        );
      }
      case "crop":
        return (
          <>
            <div className="ie-section">
              <div className="ie-sec-title">Aspect ratio</div>
              <div className="ie-seg wrap" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                {CROP_RATIOS.map((entry) => (
                  <button
                    className="ie-seg-btn"
                    data-active={ratioKey === entry.key}
                    key={entry.key}
                    onClick={() => chooseRatio(entry.key)}
                    type="button"
                  >
                    {entry.label}
                  </button>
                ))}
              </div>
              <button
                className="ie-btn block"
                data-active={rotated}
                disabled={ratioKey === "free" || ratioKey === "1:1"}
                onClick={toggleRotate}
                type="button"
              >
                ⟲ Swap orientation
              </button>
            </div>
            <div className="ie-section">
              <div className="ie-sec-title">Size</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: "8px", alignItems: "end" }}>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Width</span>
                  </div>
                  <input
                    className="ie-input ie-numfield"
                    onChange={(event) => setCropDim("width", event.target.value)}
                    type="number"
                    value={cropRect ? Math.round(cropRect.width) : ""}
                  />
                </div>
                <span style={{ paddingBottom: "10px", color: "var(--ie-faint)" }}>×</span>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Height</span>
                  </div>
                  <input
                    className="ie-input ie-numfield"
                    onChange={(event) => setCropDim("height", event.target.value)}
                    type="number"
                    value={cropRect ? Math.round(cropRect.height) : ""}
                  />
                </div>
              </div>
              <div className="ie-field">
                <div className="ie-field-top">
                  <span className="ie-field-label">Straighten</span>
                  <span className="ie-field-val">
                    {straighten > 0 ? "+" : ""}
                    {straighten}°
                  </span>
                </div>
                <input
                  className="ie-range"
                  max={15}
                  min={-15}
                  onChange={(event) => setStraighten(Number(event.target.value))}
                  type="range"
                  value={straighten}
                />
                <p className="ie-note">Rotates the image within the crop; applied on Apply. Inset the crop so the corners stay filled.</p>
              </div>
            </div>
            <div className="ie-section">
              <button className="ie-btn block primary" onClick={applyCrop} type="button">
                Apply crop
              </button>
              <button className="ie-btn block" onClick={cancelCrop} type="button">
                Cancel
              </button>
            </div>
          </>
        );
      case "upscale":
        return (
          <>
            <div className="ie-section">
              <div className="ie-sec-title">Engine</div>
              <div className="ie-cards">
                {availableUpscaleEngines.map((entry) => (
                  <button
                    className="ie-card"
                    data-active={upscaleEngine === entry.key}
                    key={entry.key}
                    onClick={() => {
                      setUpscaleEngine(entry.key);
                      if (!entry.factors.includes(upscaleFactor)) setUpscaleFactor(entry.factors[0]);
                    }}
                    type="button"
                  >
                    <span className="ie-radio" />
                    <span>
                      <span className="ie-card-name">{entry.label}</span>
                      {UPSCALE_ENGINE_DESC[entry.key] ? (
                        <span className="ie-card-desc">{UPSCALE_ENGINE_DESC[entry.key]}</span>
                      ) : null}
                    </span>
                  </button>
                ))}
              </div>
            </div>
            <div className="ie-section">
              <div className="ie-sec-title">Scale factor</div>
              <div className="ie-seg wrap two">
                {upscaleFactorsForEngine(upscaleEngine).map((value) => (
                  <button
                    className="ie-seg-btn"
                    data-active={upscaleFactor === value}
                    key={value}
                    onClick={() => setUpscaleFactor(value)}
                    type="button"
                  >
                    {value}×
                  </button>
                ))}
              </div>
              {upscaleEngineHasSoftness(upscaleEngine) ? (
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Detail recovery</span>
                    <span className="ie-field-val">{upscaleSoftness.toFixed(2)}</span>
                  </div>
                  <input
                    className="ie-range"
                    max={1}
                    min={0}
                    onChange={(event) => setUpscaleSoftness(Number(event.target.value))}
                    step={0.05}
                    type="range"
                    value={upscaleSoftness}
                  />
                  <p className="ie-note">Higher restores more texture from a degraded source; 0 stays faithful to the original.</p>
                </div>
              ) : null}
              <div className="ie-readout">
                <span className="ie-readout-k">Output size</span>
                <span className="ie-readout-v">
                  {working.width * upscaleFactor} × {working.height * upscaleFactor}
                </span>
              </div>
            </div>
            <div className="ie-section">
              <button className="ie-btn block primary" disabled={!!aiOp} onClick={runUpscale} type="button">
                Upscale image
              </button>
              <button className="ie-btn block" onClick={() => setTool("move")} type="button">
                Cancel
              </button>
            </div>
          </>
        );
      case "detail":
        return detailModels.length === 0 ? (
          <div className="ie-section">
            <p className="ie-note">No detail-capable models installed.</p>
          </div>
        ) : (
          <>
            <div className="ie-section">
              <div className="ie-sec-title">Backbone</div>
              <select className="ie-select" onChange={(event) => setDetailModel(event.target.value)} value={detailModel}>
                {detailModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label ?? model.id}
                  </option>
                ))}
              </select>
            </div>
            <div className="ie-section">
              <div className="ie-sec-title">Refinement</div>
              <div className="ie-field">
                <div className="ie-field-top">
                  <span className="ie-field-label">Detail amount</span>
                  <span className="ie-field-val">{Math.round(detailStrength * 100)}%</span>
                </div>
                <input
                  className="ie-range"
                  max={0.8}
                  min={0.3}
                  onChange={(event) => setDetailStrength(Number(event.target.value))}
                  step={0.05}
                  type="range"
                  value={detailStrength}
                />
                <p className="ie-note">Higher invents more fine texture.</p>
              </div>
              <div className="ie-field">
                <div className="ie-field-top">
                  <span className="ie-field-label">Structure lock</span>
                  <span className="ie-field-val">{Math.round(detailCnScale * 100)}%</span>
                </div>
                <input
                  className="ie-range"
                  max={1}
                  min={0.4}
                  onChange={(event) => setDetailCnScale(Number(event.target.value))}
                  step={0.05}
                  type="range"
                  value={detailCnScale}
                />
                <p className="ie-note">Higher keeps the result closer to the source composition.</p>
              </div>
            </div>
            {/* Tile ControlNet dependency (sc-2437/sc-2438): Detail can't run without it, but it's a
                separate utility download — surface it as required with a one-click install when missing,
                and gate the run. Mirrors the managed edit-LoRA CTA in the AI Edit panel. */}
            <div className="ie-section">
              <div className="ie-sec-title">Required model</div>
              {tileControlNetReady ? (
                <p className="ie-note">✨ {tileControlNet.name ?? "SDXL Tile ControlNet"} is installed and ready.</p>
              ) : (
                <>
                  <p className="ie-note">
                    The SDXL tile ControlNet (~2.5 GB) is required for Detail enhance and isn’t installed yet.
                  </p>
                  <button
                    className="ie-btn block"
                    disabled={!tileControlNet || !createModelDownloadJob || tileControlNetDownloadRequested}
                    onClick={requestTileControlNetDownload}
                    type="button"
                  >
                    {tileControlNetDownloadRequested
                      ? "Downloading…"
                      : `Download ${tileControlNet?.name ?? "SDXL Tile ControlNet"}`}
                  </button>
                </>
              )}
            </div>
            <div className="ie-section">
              <button
                className="ie-btn block primary"
                disabled={!!aiOp || !detailModel || !tileControlNetReady}
                onClick={runDetail}
                type="button"
              >
                Enhance detail
              </button>
              <button className="ie-btn block" onClick={() => setTool("move")} type="button">
                Cancel
              </button>
            </div>
          </>
        );
      case "color":
        return (
          <>
            <div className="ie-section">
              <div className="ie-seg wrap" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                {[
                  ["adjust", "Adjust"],
                  ["levels", "Levels"],
                  ["curves", "Curves"],
                ].map(([mode, label]) => (
                  <button
                    className="ie-seg-btn"
                    data-active={colorMode === mode}
                    key={mode}
                    onClick={() => setColorMode(mode)}
                    type="button"
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            {colorMode === "adjust" ? (
              <div className="ie-section">
                <div className="ie-sec-title">Tone &amp; color</div>
                {COLOR_ADJUSTMENTS.map(({ key: adjKey, label }) => (
                  <div className="ie-field" key={adjKey}>
                    <div className="ie-field-top">
                      <span className="ie-field-label">{label}</span>
                      <span className="ie-field-val">
                        {colorAdjust[adjKey] > 0 ? "+" : ""}
                        {Math.round(colorAdjust[adjKey] * 100)}
                      </span>
                    </div>
                    <input
                      className="ie-range"
                      max={1}
                      min={-1}
                      onChange={(event) => setAdjustValue(adjKey, Number(event.target.value))}
                      onDoubleClick={() => resetAdjust(adjKey)}
                      step={0.01}
                      type="range"
                      value={colorAdjust[adjKey]}
                    />
                  </div>
                ))}
              </div>
            ) : null}
            {colorMode === "levels" ? (
              <div className="ie-section">
                <div className="ie-sec-title">Channel</div>
                <select className="ie-select" onChange={(event) => setColorChannel(event.target.value)} value={colorChannel}>
                  <option value="master">Master (RGB)</option>
                  <option value="r">Red</option>
                  <option value="g">Green</option>
                  <option value="b">Blue</option>
                </select>
                <canvas className="ie-histo" height={56} ref={histogramRef} width={280} />
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Black point</span>
                    <span className="ie-field-val">{levels[colorChannel].black}</span>
                  </div>
                  <input
                    className="ie-range"
                    max={254}
                    min={0}
                    onChange={(event) => setLevelsValue("black", Number(event.target.value))}
                    step={1}
                    type="range"
                    value={levels[colorChannel].black}
                  />
                </div>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">Gamma</span>
                    <span className="ie-field-val">{levels[colorChannel].gamma.toFixed(2)}</span>
                  </div>
                  <input
                    className="ie-range"
                    max={2.5}
                    min={0.1}
                    onChange={(event) => setLevelsValue("gamma", Number(event.target.value))}
                    step={0.01}
                    type="range"
                    value={levels[colorChannel].gamma}
                  />
                </div>
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label">White point</span>
                    <span className="ie-field-val">{levels[colorChannel].white}</span>
                  </div>
                  <input
                    className="ie-range"
                    max={255}
                    min={1}
                    onChange={(event) => setLevelsValue("white", Number(event.target.value))}
                    step={1}
                    type="range"
                    value={levels[colorChannel].white}
                  />
                </div>
              </div>
            ) : null}
            {colorMode === "curves" ? (
              <div className="ie-section">
                <div className="ie-sec-title">Tone curve</div>
                <select className="ie-select" onChange={(event) => setColorChannel(event.target.value)} value={colorChannel}>
                  <option value="master">Master (RGB)</option>
                  <option value="r">Red</option>
                  <option value="g">Green</option>
                  <option value="b">Blue</option>
                </select>
                <div className="ie-curvewrap">
                  <CurveEditor
                    onChange={(points) => setCurves((prev) => ({ ...prev, [colorChannel]: points }))}
                    points={curves[colorChannel]}
                    stroke={channelStroke}
                  />
                </div>
                <p className="ie-note">Drag points to reshape the curve. Double-click to add a point.</p>
              </div>
            ) : null}
            <div className="ie-section">
              <button className="ie-btn block" disabled={activeGradeIsIdentity()} onClick={resetActiveColorMode} type="button">
                Reset
              </button>
              <button
                className="ie-btn block primary"
                disabled={activeGradeIsIdentity()}
                onClick={applyColorGrade}
                type="button"
              >
                Apply grade
              </button>
            </div>
          </>
        );
      case "edit":
        return renderEditPanel();
      case "boxes":
        return renderBoxesPanel();
      default:
        return null;
    }
  };

  const renderLoraSection = () => {
    const { selectedLoraIds, toggleLora, weightFor, setWeight } = editLoraSelection;
    // The managed image-edit LoRA is applied automatically (sc-11069) — hidden from the manual list.
    const compatibleLoras = pickerCompatibleLoras;
    const nextLora = compatibleLoras.find((lora) => !selectedLoraIds.includes(lora.id));
    const addDisabled = !nextLora || selectedLoraIds.length >= MAX_JOB_LORAS_TOTAL;
    const addHint = loraAddHint({
      selectedCount: selectedLoraIds.length,
      hasNext: Boolean(nextLora),
      max: MAX_JOB_LORAS_TOTAL,
    });
    return (
      <div className="ie-section">
        <div className="ie-sec-title">
          LoRAs
          <button
            className="ie-btn sm ghost"
            disabled={addDisabled}
            onClick={() => nextLora && toggleLora(nextLora)}
            style={{ height: "24px" }}
            type="button"
          >
            + Add
          </button>
        </div>
        {/* Why + Add is dead once you've added some (epic 10644 / sc-10653). See loraAddHint. */}
        {addHint ? <p className="ie-note">{addHint}</p> : null}
        {selectedLoraIds.length ? (
          selectedLoraIds
            .map((id) => compatibleLoras.find((lora) => lora.id === id))
            .filter(Boolean)
            .map((lora) => (
              <div className="ie-lora" key={lora.id}>
                <div className="ie-lora-top">
                  <span className="ie-lora-name">{lora.name ?? lora.id}</span>
                  <button className="ie-btn icon sm ghost" onClick={() => toggleLora(lora)} title="Remove" type="button">
                    ✕
                  </button>
                </div>
                <LoraKeywordSummary lora={lora} />
                <div className="ie-field">
                  <div className="ie-field-top">
                    <span className="ie-field-label" style={{ fontSize: "11.5px", color: "var(--ie-muted)" }}>
                      Weight
                    </span>
                    <span className="ie-field-val">{weightFor(lora).toFixed(2)}</span>
                  </div>
                  <input
                    aria-label={`${lora.name ?? lora.id} weight`}
                    className="ie-range"
                    max={LORA_WEIGHT_MAX}
                    min={LORA_WEIGHT_MIN}
                    onChange={(event) => setWeight(lora.id, Number(event.target.value))}
                    step={LORA_WEIGHT_STEP}
                    type="range"
                    value={weightFor(lora)}
                  />
                </div>
              </div>
            ))
        ) : (
          <p className="ie-note">No LoRAs applied. Add a style or subject LoRA to steer the edit.</p>
        )}
      </div>
    );
  };

  const renderEditPanel = () => {
    if (editModels.length === 0) {
      return (
        <div className="ie-section">
          <p className="ie-note">No edit-capable models installed.</p>
        </div>
      );
    }
    return (
      <>
        <div className="ie-section">
          <div className="ie-sec-title">Model</div>
          <select className="ie-select" onChange={(event) => setEditModel(event.target.value)} value={editModel}>
            {editModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label ?? model.id}
              </option>
            ))}
          </select>
          <div className="ie-field">
            <div className="ie-field-top">
              <span className="ie-field-label">Instruction</span>
            </div>
            <textarea
              className="ie-textarea"
              onChange={(event) => setEditPrompt(event.target.value)}
              placeholder="Describe the edit — e.g. “replace the background with a foggy pine forest at dawn”"
              value={editPrompt}
            />
          </div>
        </div>

        {/* Managed image-edit LoRA (epic 10871, sc-11069): auto-applied for the user — a status note
            when installed, a one-click download that gates the run when not. Inert for edit models
            that need none. Placed right under the model so it reads as part of the edit surface. */}
        {editLora ? (
          editLoraInstalled ? (
            <div className="ie-section">
              <p className="ie-note">✨ {editLora.name} is applied automatically for editing.</p>
              {/* Identity strength (sc-11798): the managed edit LoRA is hidden from the manual
                  picker, so expose its apply weight here — threaded into buildEditJobBody's
                  editLoraWeight → the payload edit-LoRA `weight`. Higher = stronger conditioning. */}
              <div className="lora-slot-weight edit-lora-strength">
                <label>
                  <span>Identity strength</span>
                  <span className="lora-slot-weight-value">
                    {editLoraSelection.weightFor(editLora).toFixed(2)}
                  </span>
                </label>
                <input
                  aria-label={`${editLora.name} identity strength`}
                  max="2"
                  min="0"
                  onChange={(event) =>
                    editLoraSelection.setWeight(editLora.id, Number(event.target.value))
                  }
                  step="0.05"
                  type="range"
                  value={editLoraSelection.weightFor(editLora)}
                />
              </div>
            </div>
          ) : (
            <div className="ie-section">
              <p className="ie-note">{editLora.name} is required to edit — the base can’t edit without it.</p>
              <button
                className="ie-btn block"
                disabled={editLoraDownloadRequested}
                onClick={requestEditLoraDownload}
                type="button"
              >
                {editLoraDownloadRequested ? "Downloading…" : `Download ${editLora.name}`}
              </button>
            </div>
          )
        ) : null}

        {pickerCompatibleLoras.length ? renderLoraSection() : null}

        <div className="ie-section">
          <div className="ie-sec-title">Output</div>
          <div className="ie-field">
            <span className="ie-field-label" style={{ marginBottom: "2px" }}>
              Aspect
            </span>
            <div className="ie-seg wrap four">
              {EDIT_OUTPUT_ASPECTS.map((aspect) => (
                <button
                  className="ie-seg-btn"
                  data-active={editAspect === aspect.key}
                  key={aspect.key}
                  onClick={() => setEditAspect(aspect.key)}
                  type="button"
                >
                  {aspect.label}
                </button>
              ))}
            </div>
          </div>
          {editAspect !== "match" ? (
            <FitModeControl
              inpaintCapable={canMask}
              label="Fill new area"
              onChange={setEditFitMode}
              value={effectiveFitMode(editFitMode, canMask)}
            />
          ) : null}
        </div>

        {canMask ? (
          <div className="ie-section">
            <div className="ie-sec-title">
              Mask
              <button
                className="ie-btn sm ghost"
                data-active={maskMode}
                onClick={() => setMaskMode((on) => !on)}
                style={{ height: "24px" }}
                type="button"
              >
                {maskMode ? "On" : "Off"}
              </button>
            </div>
            {maskMode ? (
              <>
                {smartSelectSupported ? (
                  <div className="ie-seg two" style={{ width: "100%" }}>
                    <button
                      className="ie-seg-btn"
                      data-active={maskSubTool === "brush"}
                      onClick={() => setMaskSubTool("brush")}
                      type="button"
                    >
                      Brush
                    </button>
                    <button
                      className="ie-seg-btn"
                      data-active={maskSubTool === "select"}
                      disabled={aiOp?.label === "smart select"}
                      onClick={() => {
                        setMaskSubTool("select");
                        setMaskErase(false);
                      }}
                      type="button"
                    >
                      {aiOp?.label === "smart select" ? "Segmenting…" : "Smart select"}
                    </button>
                  </div>
                ) : null}
                {!smartSelectSupported || maskSubTool === "brush" ? (
                  <>
                    <div className="ie-field">
                      <div className="ie-field-top">
                        <span className="ie-field-label">Brush size</span>
                        <span className="ie-field-val">{maskBrush} px</span>
                      </div>
                      <input
                        className="ie-range"
                        max={300}
                        min={5}
                        onChange={(event) => setMaskBrush(Number(event.target.value))}
                        step={1}
                        type="range"
                        value={maskBrush}
                      />
                    </div>
                    <button className="ie-btn block" data-active={maskErase} onClick={() => setMaskErase((on) => !on)} type="button">
                      Eraser
                    </button>
                  </>
                ) : (
                  <p className="ie-note">Drag a box around an object on the canvas — SAM3 auto-masks it.</p>
                )}
                <div>
                  <span className="ie-field-label" style={{ display: "block", marginBottom: "7px" }}>
                    Refine selection
                  </span>
                  <div className="ie-field" style={{ marginBottom: "8px" }}>
                    <div className="ie-field-top">
                      <span className="ie-field-label" style={{ fontSize: "11.5px", color: "var(--ie-muted)" }}>
                        Radius
                      </span>
                      <span className="ie-field-val">{maskRefineRadius}px</span>
                    </div>
                    <input
                      className="ie-range"
                      max={40}
                      min={1}
                      onChange={(event) => setMaskRefineRadius(Number(event.target.value))}
                      step={1}
                      type="range"
                      value={maskRefineRadius}
                    />
                  </div>
                  <div className="ie-chip-row">
                    <button
                      className="ie-chip"
                      disabled={!maskHasContent(maskLines) && !maskBaseImage}
                      onClick={() => refineMask("feather")}
                      type="button"
                    >
                      Feather
                    </button>
                    <button
                      className="ie-chip"
                      disabled={!maskHasContent(maskLines) && !maskBaseImage}
                      onClick={() => refineMask("grow")}
                      type="button"
                    >
                      Grow
                    </button>
                    <button
                      className="ie-chip"
                      disabled={!maskHasContent(maskLines) && !maskBaseImage}
                      onClick={() => refineMask("shrink")}
                      type="button"
                    >
                      Shrink
                    </button>
                    <button className="ie-chip" onClick={() => refineMask("invert")} type="button">
                      Invert
                    </button>
                    <button
                      className="ie-chip"
                      disabled={!maskLines.length && !maskBaseImage}
                      onClick={clearMask}
                      type="button"
                    >
                      Clear
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <p className="ie-note">Turn on a mask to confine the edit to a painted or selected region (inpaint).</p>
            )}
          </div>
        ) : null}

        {multiRefCapable ? (
          <div className="ie-section">
            <div className="ie-sec-title">Reference images</div>
            <div className="ie-refs">
              {refAssetIds.map((id) => {
                const asset = imageAssets.find((item) => item.id === id);
                return (
                  <div className="ie-ref" key={id}>
                    {asset ? <img alt="" src={assetUrl(asset)} /> : <span>?</span>}
                    <button
                      aria-label="Remove reference"
                      className="ie-ref-remove"
                      onClick={() => setRefAssetIds((prev) => prev.filter((other) => other !== id))}
                      type="button"
                    >
                      ✕
                    </button>
                  </div>
                );
              })}
              <button
                className="ie-ref-add"
                disabled={refAssetIds.length >= MAX_EDIT_REFERENCES - 1}
                onClick={() => setRefPickerOpen(true)}
                title="Condition the edit on reference image(s)"
                type="button"
              >
                +
              </button>
            </div>
          </div>
        ) : null}

        <div className="ie-section">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
            <div className="ie-field">
              <span className="ie-field-label" style={{ marginBottom: "2px" }}>
                Seed
              </span>
              <input
                className="ie-input"
                min={0}
                onChange={(event) => setEditSeed(event.target.value)}
                placeholder="Random"
                style={{ fontFamily: "var(--ie-mono)" }}
                type="number"
                value={editSeed}
              />
            </div>
            <div className="ie-field">
              <span className="ie-field-label" style={{ marginBottom: "2px" }}>
                Guidance
              </span>
              <input
                className="ie-input"
                min={0}
                onChange={(event) => setEditGuidance(event.target.value)}
                placeholder={guidanceDefaultFromModel(selectedEditModel)?.toString() ?? "Default"}
                step={0.1}
                style={{ fontFamily: "var(--ie-mono)" }}
                type="number"
                value={editGuidance}
              />
            </div>
          </div>
          <button
            className="ie-btn block primary"
            disabled={!editPrompt.trim() || !!aiOp || editLoraRequiredMissing}
            onClick={runEdit}
            type="button"
          >
            {maskActive ? "Inpaint region" : "Generate edit"}
          </button>
        </div>
      </>
    );
  };

  const renderBoxesPanel = () => (
    <>
      <div className="ie-section">
        <div className="ie-sec-title">Box color</div>
        <div className="ie-swatches">
          {BOX_PALETTE.map((entry) => (
            <button
              aria-label={entry.name}
              className="ie-swatch"
              data-active={boxColor === entry.value}
              key={entry.value}
              onClick={() => chooseBoxColor(entry.value)}
              style={{ background: entry.value }}
              title={entry.name}
              type="button"
            />
          ))}
          <label className="ie-swatch ie-swatch-custom" title="Custom color">
            <input aria-label="Custom box color" onChange={(event) => chooseBoxColor(event.target.value)} type="color" value={boxColor.toLowerCase()} />
          </label>
        </div>
        <p className="ie-note">Drag on the canvas to draw a colored region, then describe what belongs there.</p>
      </div>

      <div className="ie-section">
        <div className="ie-sec-title">Regions ({boxes.length})</div>
        {boxes.length ? (
          <div className="ie-chip-row">
            {boxes.map((box, index) => {
              const incomplete = boxMetadataGaps(box).length > 0;
              return (
                <button
                  className="ie-chip"
                  data-active={selectedBoxId === box.id}
                  key={box.id}
                  onClick={() => setSelectedBoxId(box.id)}
                  title={box.desc ? `${index + 1}: ${box.desc}` : `Box ${index + 1} — needs a description`}
                  type="button"
                >
                  <span className="ie-dot" style={{ background: box.color }} />
                  {index + 1}
                  {incomplete ? <span className="warn">!</span> : null}
                </button>
              );
            })}
          </div>
        ) : (
          <p className="ie-note">Drag on the image to draw a box.</p>
        )}
        {boxes.length ? (
          <div style={{ display: "flex", gap: "8px" }}>
            <button className="ie-btn sm" disabled={!selectedBoxId} onClick={() => deleteBox(selectedBoxId)} type="button">
              Delete
            </button>
            <button className="ie-btn sm" disabled={!boxes.length} onClick={clearBoxes} type="button">
              Clear all
            </button>
          </div>
        ) : null}
      </div>

      {selectedBox ? (
        <div className="ie-section">
          <div className="ie-sec-title">
            Region details
            <span className="ie-field-val">Box {boxes.indexOf(selectedBox) + 1}</span>
          </div>
          <div className="ie-seg two" style={{ width: "100%" }}>
            <button
              className="ie-seg-btn"
              data-active={selectedBox.type === "obj"}
              onClick={() => updateBox(selectedBox.id, { type: "obj" })}
              type="button"
            >
              Object
            </button>
            <button
              className="ie-seg-btn"
              data-active={selectedBox.type === "text"}
              onClick={() => updateBox(selectedBox.id, { type: "text" })}
              type="button"
            >
              Text
            </button>
          </div>
          <div className="ie-field">
            <span className="ie-field-label" style={{ marginBottom: "2px" }}>
              Description
            </span>
            <input
              className="ie-input"
              onChange={(event) => updateBox(selectedBox.id, { desc: event.target.value })}
              placeholder="What is in this region?"
              value={selectedBox.desc ?? ""}
            />
          </div>
          {selectedBox.type === "text" ? (
            <div className="ie-field">
              <span className="ie-field-label" style={{ marginBottom: "2px" }}>
                Literal text
              </span>
              <input
                className="ie-input"
                onChange={(event) => updateBox(selectedBox.id, { text: event.target.value })}
                placeholder="Text to render"
                value={selectedBox.text ?? ""}
              />
            </div>
          ) : null}
          <div className="ie-field">
            <span className="ie-field-label" style={{ marginBottom: "2px" }}>
              Element colors ({(selectedBox.colorPalette ?? []).length}/{MAX_BOX_PALETTE})
            </span>
            <div className="ie-swatches">
              {(selectedBox.colorPalette ?? []).map((color) => (
                <button
                  aria-label={`Remove ${color}`}
                  className="ie-swatch"
                  key={color}
                  onClick={() => updateBox(selectedBox.id, { colorPalette: removePaletteColor(selectedBox.colorPalette, color) })}
                  style={{ background: color }}
                  title={`Remove ${color}`}
                  type="button"
                />
              ))}
              {(selectedBox.colorPalette ?? []).length < MAX_BOX_PALETTE ? (
                <label className="ie-swatch ie-swatch-custom" title="Add color">
                  <input
                    aria-label="Add element color"
                    onChange={(event) => updateBox(selectedBox.id, { colorPalette: addPaletteColor(selectedBox.colorPalette, event.target.value) })}
                    type="color"
                  />
                </label>
              ) : null}
            </div>
          </div>
          {selectedBoxGaps.length ? (
            <p className="ie-note">For Ideogram layout this box still needs {selectedBoxGaps.join(", ")}. The color-keyed edit path only needs a color + description.</p>
          ) : (
            <p className="ie-note" style={{ color: "var(--ie-accent)" }}>
              Ready for Ideogram layout ✓
            </p>
          )}
        </div>
      ) : null}

      {boxes.length ? (
        <div className="ie-section">
          <div className="ie-sec-title">Generate</div>
          {editModels.length ? (
            <>
              <select className="ie-select" onChange={(event) => setEditModel(event.target.value)} value={editModel}>
                {editModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label ?? model.id}
                  </option>
                ))}
              </select>
              <input
                className="ie-input"
                onChange={(event) => setEditPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !aiOp && editModel) runBoxEdit();
                }}
                placeholder="Prompt (or use Auto-prompt)"
                value={editPrompt}
              />
              <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "8px" }}>
                <button className="ie-btn" disabled={!composedPrompt} onClick={() => setEditPrompt(composedPrompt)} type="button">
                  Auto-prompt
                </button>
                <button className="ie-btn primary" disabled={!!aiOp || !editModel} onClick={runBoxEdit} type="button">
                  Generate
                </button>
              </div>
            </>
          ) : (
            <p className="ie-note">No edit-capable models installed.</p>
          )}
        </div>
      ) : null}
    </>
  );

  const handleLayerOpacityInput = (id, raw) => {
    const start = !layerOpacityGestureRef.current;
    layerOpacityGestureRef.current = true;
    changeLayerOpacity(id, Math.max(0, Math.min(100, Number(raw) || 0)) / 100, start);
  };
  const endLayerOpacityGesture = () => {
    layerOpacityGestureRef.current = false;
  };

  const renderLayers = () => (
    <aside className="ie-layers" aria-label="Layers">
      <div className="ie-layers-head">
        <button className="ie-layers-title" onClick={() => setLayersOpen((v) => !v)} title="Collapse layers" type="button">
          <span className="ie-acc-chev" data-open={layersOpen}>
            <IeChevron />
          </span>
          <svg fill="none" height="15" stroke="currentColor" strokeLinejoin="round" strokeWidth={2} viewBox="0 0 24 24" width="15">
            <polygon points="12 2 2 7 12 12 22 7 12 2" />
            <polyline points="2 17 12 22 22 17" />
            <polyline points="2 12 12 17 22 12" />
          </svg>
          Layers
          <span className="ie-layers-count">{layerCount}</span>
        </button>
        <button className="ie-btn icon sm ghost" disabled={Boolean(aiOp)} onClick={addBlankLayer} title="Add layer" type="button">
          +
        </button>
      </div>
      {layersOpen ? (
        <div className="ie-layers-list">
          {working.layers
            .map((layer, index) => ({ layer, index }))
            .reverse()
            .map(({ layer, index }) => {
              const isActive = layer.id === working.activeLayerId;
              const pct = Math.round(layer.opacity * 100);
              return (
                <div className="ie-layer" data-active={isActive} key={layer.id} onClick={() => selectLayer(layer.id)}>
                  <div className="ie-layer-row">
                    <button
                      className="ie-layer-vis"
                      onClick={(event) => {
                        event.stopPropagation();
                        toggleLayerVisible(layer.id);
                      }}
                      title="Toggle visibility"
                      type="button"
                    >
                      {layer.visible ? <IeEyeOpen /> : <IeEyeOff />}
                    </button>
                    {layer.objectUrl ? (
                      <img
                        alt=""
                        className="ie-layer-thumb"
                        src={layer.objectUrl}
                        style={layer.visible ? undefined : { opacity: 0.35, filter: "grayscale(1)" }}
                      />
                    ) : (
                      <span className="ie-layer-thumb" />
                    )}
                    <span
                      className="ie-layer-name"
                      onDoubleClick={() => {
                        const name = window.prompt("Rename layer", layer.name)?.trim();
                        if (name) renameLayer(layer.id, name);
                      }}
                      title="Double-click to rename"
                    >
                      {layer.name}
                    </span>
                    {layer.blendMode && layer.blendMode !== "source-over" ? (
                      <span className="ie-layer-blend">
                        {(BLEND_MODES.find((mode) => mode.value === layer.blendMode)?.label ?? layer.blendMode).slice(0, 4)}
                      </span>
                    ) : null}
                  </div>
                  {isActive ? (
                    <>
                      <div className="ie-layer-op" onClick={(event) => event.stopPropagation()}>
                        <input
                          className="ie-range"
                          max={100}
                          min={0}
                          onBlur={endLayerOpacityGesture}
                          onChange={(event) => handleLayerOpacityInput(layer.id, event.target.value)}
                          onMouseUp={endLayerOpacityGesture}
                          onTouchEnd={endLayerOpacityGesture}
                          type="range"
                          value={pct}
                        />
                        <div className="ie-layer-opnum">
                          <input
                            aria-label={`${layer.name} opacity`}
                            className="ie-input"
                            max={100}
                            min={0}
                            onBlur={endLayerOpacityGesture}
                            onChange={(event) => handleLayerOpacityInput(layer.id, event.target.value)}
                            type="number"
                            value={pct}
                          />
                        </div>
                      </div>
                      <div className="ie-layer-blendsel" onClick={(event) => event.stopPropagation()}>
                        <select
                          aria-label={`${layer.name} blend mode`}
                          className="ie-select"
                          onChange={(event) => setLayerBlend(layer.id, event.target.value)}
                          value={layer.blendMode || "source-over"}
                        >
                          {BLEND_MODES.map((mode) => (
                            <option key={mode.value} value={mode.value}>
                              {mode.label}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="ie-chip-row" onClick={(event) => event.stopPropagation()} style={{ marginTop: "8px" }}>
                        <button className="ie-btn sm ghost" disabled={index >= working.layers.length - 1} onClick={() => reorderLayer(layer.id, index + 1)} title="Move up" type="button">
                          ↑
                        </button>
                        <button className="ie-btn sm ghost" disabled={index <= 0} onClick={() => reorderLayer(layer.id, index - 1)} title="Move down" type="button">
                          ↓
                        </button>
                        <button className="ie-btn sm ghost" onClick={() => duplicateLayerById(layer.id)} title="Duplicate" type="button">
                          ⧉
                        </button>
                        <button className="ie-btn sm ghost danger" disabled={working.layers.length <= 1} onClick={() => deleteLayer(layer.id)} title="Delete" type="button">
                          ✕
                        </button>
                      </div>
                    </>
                  ) : null}
                </div>
              );
            })}
        </div>
      ) : null}
    </aside>
  );

  const renderInspectorBody = () => (
    <div className="ie-insp-body">
      {EDITOR_TOOL_ORDER.map((key) => {
        const open = tool === key && panelOpen;
        return (
          <React.Fragment key={key}>
            <button
              className="ie-acc-head"
              data-active={tool === key}
              disabled={toolIsDisabled(key)}
              onClick={() => onAccordionHead(key)}
              type="button"
            >
              <span className="ie-acc-ic">{EDITOR_TOOL_ICONS[key]}</span>
              <span className="ie-acc-label">{EDITOR_TOOL_META[key].label}</span>
              <span className="ie-acc-chev" data-open={open}>
                <IeChevron />
              </span>
            </button>
            {open ? renderToolPanel(key) : null}
          </React.Fragment>
        );
      })}
    </div>
  );

  return (
    <section className="image-editor-surface ie-shell" data-ie-layout={layout}>
      <header className="ie-topbar">
        <div className="ie-brand">
          <div className="ie-brand-mark">
            <svg fill="none" height="15" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.2} viewBox="0 0 24 24" width="15">
              <path d="M7 2v15a1 1 0 001 1h15M2 7h15a1 1 0 011 1v15" />
            </svg>
          </div>
          <div className="ie-doc">
            <div className="ie-doc-name" title={docName}>
              {docName}
            </div>
            <div className="ie-doc-sub">{docSub}</div>
          </div>
        </div>

        <button className="ie-btn sm" onClick={() => setPickerOpen(true)} type="button">
          Open
        </button>
        <button
          className="ie-btn sm"
          onClick={() => setNewLayoutOpen(true)}
          title="Start a blank canvas for box layout"
          type="button"
        >
          New layout
        </button>
        {working && working.source.assetId ? (
          <button
            className="ie-btn sm ghost"
            onClick={() => setPreviewAsset?.(imageAssets.find((item) => item.id === working.source.assetId))}
            title="Preview the source asset"
            type="button"
          >
            Source
          </button>
        ) : null}
        <button
          aria-pressed={shortcutsOpen}
          className="ie-btn icon sm ghost"
          onClick={() => setShortcutsOpen((on) => !on)}
          title="Keyboard shortcuts (?)"
          type="button"
        >
          ⌨
        </button>

        <div className="ie-spacer" />

        {working ? (
          <div className="ie-topgroup">
            <button className="ie-btn icon sm" disabled={!historyFlags.canUndo || Boolean(aiOp)} onClick={undo} title="Undo (⌘Z)" type="button">
              <svg fill="none" height="15" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} viewBox="0 0 24 24" width="15">
                <path d="M9 14L4 9l5-5" />
                <path d="M4 9h11a5 5 0 015 5v0a5 5 0 01-5 5H9" />
              </svg>
            </button>
            <button className="ie-btn icon sm" disabled={!historyFlags.canRedo || Boolean(aiOp)} onClick={redo} title="Redo (⇧⌘Z / Ctrl+Y)" type="button">
              <svg fill="none" height="15" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} viewBox="0 0 24 24" width="15">
                <path d="M15 14l5-5-5-5" />
                <path d="M20 9H9a5 5 0 00-5 5v0a5 5 0 005 5h6" />
              </svg>
            </button>
          </div>
        ) : null}

        <div className="ie-divider" />

        <div className="ie-seg" title="Panel layout">
          {[
            ["accordion", "Stacked panels", <React.Fragment key="g"><rect height="16" rx="2" width="18" x="3" y="4" /><line x1="3" x2="21" y1="9" y2="9" /><line x1="3" x2="21" y1="14" y2="14" /></React.Fragment>],
            ["right", "Inspector right", <React.Fragment key="g"><rect height="16" rx="2" width="18" x="3" y="4" /><line x1="15" x2="15" y1="4" y2="20" /></React.Fragment>],
            ["left", "Inspector left", <React.Fragment key="g"><rect height="16" rx="2" width="18" x="3" y="4" /><line x1="9" x2="9" y1="4" y2="20" /></React.Fragment>],
            ["bottom", "Dock bottom", <React.Fragment key="g"><rect height="16" rx="2" width="18" x="3" y="4" /><line x1="3" x2="21" y1="14" y2="14" /></React.Fragment>],
          ].map(([mode, label, glyph]) => (
            <button
              className="ie-seg-btn ie-seg-icon"
              data-active={layout === mode}
              key={mode}
              onClick={() => setLayout(mode)}
              title={label}
              type="button"
            >
              <svg fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                {glyph}
              </svg>
            </button>
          ))}
        </div>

        <button
          className="ie-btn icon sm ghost"
          onClick={() => changeTheme?.(theme === "dark" ? "light" : "dark")}
          title="Toggle theme"
          type="button"
        >
          {theme === "dark" ? "☀" : "☾"}
        </button>

        {working ? (
          <>
            <div className="ie-divider" />
            {/* Unsaved-edits indicator (sc-11968): a pill while the working doc has edits not
                yet saved to the Library, swapped for the "Saved ✓" hint once a Save lands. */}
            {saveStatusIndicator({ dirty, savedAssetId }) === "unsaved" ? (
              <span className="ie-unsaved-badge" role="status" title="You have unsaved edits">
                <span className="ie-unsaved-dot" aria-hidden="true" />
                Unsaved
              </span>
            ) : null}
            {saveStatusIndicator({ dirty, savedAssetId }) === "saved" ? (
              <span className="ie-doc-sub" style={{ color: "var(--ie-accent)" }}>
                Saved ✓
              </span>
            ) : null}
            <button className="ie-btn sm" onClick={runDownload} title="Download a PNG to your computer" type="button">
              Download
            </button>
            <button
              className="ie-btn sm primary"
              disabled={!dirty || saving}
              onClick={runSave}
              title="Save a new image to the project Library"
              type="button"
            >
              {saving ? "Saving…" : "Save"}
            </button>
            {/* Explicit Close/Discard (sc-11968): intentionally clears the working doc.
                Guarded by the desktop-safe confirm when there are unsaved edits / a running op. */}
            <button
              className="ie-btn sm ghost danger"
              onClick={closeDoc}
              title="Close this image (discard unsaved edits)"
              type="button"
            >
              Close
            </button>
          </>
        ) : null}
      </header>

      {working ? (
        <nav className="ie-rail" aria-label="Tools">
          <div className="ie-rail-cap">Tools</div>
          {EDITOR_TOOL_ORDER.map((key) => (
            <button
              className="ie-tool"
              data-active={tool === key}
              disabled={toolIsDisabled(key)}
              key={key}
              onClick={() => selectTool(key)}
              title={EDITOR_TOOL_META[key].desc}
              type="button"
            >
              {EDITOR_TOOL_ICONS[key]}
              <span>{EDITOR_TOOL_META[key].label}</span>
            </button>
          ))}
        </nav>
      ) : null}

      <main className="ie-canvas" onDragOver={(event) => event.preventDefault()} onDrop={handleDrop} ref={containerRef}>
        {status.error ? (
          <div
            className="ie-hint"
            role="alert"
            style={{ borderColor: "color-mix(in srgb, var(--ie-danger) 55%, var(--ie-border))", color: "var(--ie-danger)" }}
          >
            {status.error}
          </div>
        ) : working ? (
          <div className="ie-hint">
            {EDITOR_TOOL_ICONS[tool]}
            <span>{toolHint}</span>
          </div>
        ) : null}
        {working && stageSize.width > 0 && stageSize.height > 0 ? (
          <Stage
            draggable={tool !== "crop" && tool !== "boxes" && tool !== "transform" && !maskMode}
            height={stageSize.height}
            onDragEnd={(event) => {
              if (event.target !== event.target.getStage()) return;
              const stage = event.target.getStage();
              setView((prev) => ({ ...prev, x: stage.x(), y: stage.y() }));
            }}
            onMouseDown={handleStagePointerDown}
            onMouseMove={handleStagePointerMove}
            onMouseUp={handleStagePointerUp}
            onTouchStart={handleStagePointerDown}
            onTouchMove={handleStagePointerMove}
            onTouchEnd={handleStagePointerUp}
            onWheel={handleWheel}
            scaleX={view.scale}
            scaleY={view.scale}
            width={stageSize.width}
            x={view.x}
            y={view.y}
          >
            <Layer>
              <Rect
                fill="#ffffff"
                height={working.height}
                name="editor-bg"
                shadowBlur={12}
                shadowColor="rgba(0,0,0,0.35)"
                width={working.width}
                x={0}
                y={0}
              />
              {/* Editor layers (sc-6117): one <KonvaImage> per raster layer, bottom→top,
                  honoring per-layer visibility / opacity / blend / transform. The ACTIVE
                  layer carries the color-grade filter + the cached node ref (the live
                  preview) — multi-layer creation + selection arrive with sc-6118/6119. */}
              {working.layers.map((layer) => {
                const isActive = layer.id === working.activeLayerId;
                const t = layer.transform;
                return (
                  <KonvaImage
                    key={layer.id}
                    globalCompositeOperation={layer.blendMode}
                    height={layer.image.naturalHeight}
                    image={layer.image}
                    name="editor-image"
                    opacity={layer.opacity}
                    rotation={t.rotation}
                    scaleX={t.scaleX}
                    scaleY={t.scaleY}
                    visible={layer.visible}
                    width={layer.image.naturalWidth}
                    x={t.x}
                    y={t.y}
                    {...(isActive
                      ? {
                          colorAdjust,
                          gradeMode: colorMode,
                          gradeLevels: levels,
                          gradeCurves: curves,
                          filters: [konvaColorFilter],
                          ref: imageNodeRef,
                        }
                      : {})}
                    {...(isActive && tool === "transform"
                      ? { draggable: true, onDragEnd: commitActiveLayerTransform, onTransformEnd: commitActiveLayerTransform }
                      : {})}
                  />
                );
              })}
              {tool === "transform" ? (
                // Per-layer transform (sc-6120): move / scale / rotate the active layer.
                <Transformer anchorSize={8} borderStroke="#ffffff" ref={layerTransformerRef} rotateEnabled />
              ) : null}
              {tool === "crop" && cropRect ? (
                <>
                  {cropOverlayRects(working.width, working.height, cropRect).map((rect, index) => (
                    <Rect
                      key={index}
                      fill="rgba(0,0,0,0.55)"
                      height={rect.height}
                      listening={false}
                      width={rect.width}
                      x={rect.x}
                      y={rect.y}
                    />
                  ))}
                  <Rect
                    draggable
                    fill="rgba(255,255,255,0.01)"
                    height={cropRect.height}
                    onDragEnd={handleCropDragEnd}
                    onTransformEnd={handleCropTransformEnd}
                    ref={cropRectRef}
                    stroke="#ffffff"
                    strokeScaleEnabled={false}
                    strokeWidth={2}
                    width={cropRect.width}
                    x={cropRect.x}
                    y={cropRect.y}
                  />
                  <Transformer
                    anchorSize={8}
                    borderStroke="#ffffff"
                    boundBoxFunc={(oldBox, newBox) =>
                      newBox.width < MIN_CROP_PX || newBox.height < MIN_CROP_PX ? oldBox : newBox
                    }
                    enabledAnchors={
                      ratioKey === "free"
                        ? ["top-left", "top-center", "top-right", "middle-left", "middle-right", "bottom-left", "bottom-center", "bottom-right"]
                        : ["top-left", "top-right", "bottom-left", "bottom-right"]
                    }
                    keepRatio={ratioKey !== "free"}
                    ref={transformerRef}
                    rotateEnabled={false}
                  />
                </>
              ) : null}
            </Layer>
            {canMask && (maskLines.length || maskOverlay) ? (
              // Isolated layer so the eraser's destination-out clears only the mask
              // overlay, never the image beneath it. The smart-select base (sc-3751)
              // renders first, with the brush strokes (and their erases) composited on top.
              <Layer listening={false}>
                {maskOverlay ? (
                  <KonvaImage height={working.height} image={maskOverlay} width={working.width} x={0} y={0} />
                ) : null}
                {maskLines.map((line, index) => (
                  <Line
                    globalCompositeOperation={line.erase ? "destination-out" : "source-over"}
                    key={index}
                    lineCap="round"
                    lineJoin="round"
                    points={line.points}
                    stroke="rgba(255,40,120,0.5)"
                    strokeWidth={line.size}
                  />
                ))}
              </Layer>
            ) : null}
            {tool === "edit" && maskMode && maskSubTool === "select" && selectDraft ? (
              // Live smart-select box preview (sc-3751), image-pixel coords like the crop rect.
              <Layer listening={false}>
                <Rect
                  dash={[8, 6]}
                  fill="rgba(255,40,120,0.12)"
                  height={selectDraft.height}
                  stroke="rgba(255,40,120,0.9)"
                  strokeWidth={2 / view.scale}
                  width={selectDraft.width}
                  x={selectDraft.x}
                  y={selectDraft.y}
                />
              </Layer>
            ) : null}
            {tool === "boxes" ? (
              // Box layout overlay (sc-6090): colored rects + a transformer on the
              // selected box + the dashed live-draw preview. Image-pixel coords, so
              // it pans/zooms with the canvas like the crop rect and mask.
              <Layer>
                {boxes.map((box) => (
                  <Rect
                    draggable
                    fill={boxFillStyle(box.color, 0.18)}
                    height={box.rect.height}
                    key={box.id}
                    name="layout-box"
                    onClick={() => setSelectedBoxId(box.id)}
                    onDragEnd={(event) => handleBoxDragEnd(box.id, event)}
                    onMouseDown={() => setSelectedBoxId(box.id)}
                    onTap={() => setSelectedBoxId(box.id)}
                    onTransformEnd={(event) => handleBoxTransformEnd(box.id, event)}
                    ref={(node) => registerBoxNode(box.id, node)}
                    stroke={box.color}
                    strokeScaleEnabled={false}
                    strokeWidth={selectedBoxId === box.id ? 3 : 2}
                    width={box.rect.width}
                    x={box.rect.x}
                    y={box.rect.y}
                  />
                ))}
                {boxDraft ? (
                  <Rect
                    dash={[6, 4]}
                    fill={boxFillStyle(boxColor, 0.18)}
                    height={boxDraft.height}
                    listening={false}
                    stroke={boxColor}
                    strokeScaleEnabled={false}
                    strokeWidth={2}
                    width={boxDraft.width}
                    x={boxDraft.x}
                    y={boxDraft.y}
                  />
                ) : null}
                <Transformer
                  anchorSize={8}
                  borderStroke="#ffffff"
                  boundBoxFunc={(oldBox, newBox) =>
                    newBox.width < MIN_BOX_PX || newBox.height < MIN_BOX_PX ? oldBox : newBox
                  }
                  ref={boxTransformerRef}
                  rotateEnabled={false}
                />
              </Layer>
            ) : null}
          </Stage>
        ) : (
          <div className="ie-canvas-empty">
            {status.loading ? (
              <p>Loading image…</p>
            ) : (
              <>
                <p className="ie-canvas-empty-title">Open an image to start editing</p>
                <p className="ie-note">Drag &amp; drop an image here, or click Open.</p>
                <p className="ie-note">
                  Or{" "}
                  <button className="ie-linkbtn" onClick={() => setNewLayoutOpen(true)} type="button">
                    start a blank layout
                  </button>{" "}
                  to compose with boxes.
                </p>
              </>
            )}
          </div>
        )}

        {shortcutsOpen ? (
          <div className="image-editor-shortcuts" role="dialog" aria-label="Keyboard shortcuts">
            <div className="image-editor-shortcuts-head">
              <span>Keyboard shortcuts</span>
              <button onClick={() => setShortcutsOpen(false)} title="Close (Esc)" type="button">
                ✕
              </button>
            </div>
            <div className="image-editor-shortcuts-body">
              {EDITOR_SHORTCUTS.map((section) => (
                <div className="image-editor-shortcuts-group" key={section.group}>
                  <h4>{section.group}</h4>
                  {section.items.map((item) => (
                    <div className="image-editor-shortcut-row" key={item.label}>
                      <span className="image-editor-shortcut-keys">
                        {item.keys.map((cap) => (
                          <kbd key={cap}>{cap}</kbd>
                        ))}
                      </span>
                      <span className="image-editor-shortcut-label">{item.label}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {aiOp ? (
          <div className="ie-busy">
            <div className="ie-busy-card">
              <p className="ie-busy-title">
                {aiOp.label === "upscale"
                  ? "Upscaling…"
                  : aiOp.label === "edit"
                    ? "Running AI edit…"
                    : aiOp.label === "detail"
                      ? "Enhancing detail…"
                      : aiOp.label === "smart select"
                        ? "Segmenting…"
                        : "Working…"}
              </p>
              <p className="ie-busy-msg">
                {activeAiJob?.message ||
                  (activeAiJob?.status === "queued" ? "Queued — waiting for a worker." : "Processing on GPU worker…")}
              </p>
              <div className="ie-busy-track">
                {typeof activeAiJob?.progress === "number" ? (
                  <div className="ie-busy-fill determinate" style={{ width: `${Math.round(activeAiJob.progress * 100)}%` }} />
                ) : (
                  <div className="ie-busy-fill" />
                )}
              </div>
            </div>
          </div>
        ) : null}

        {working ? (
          <div className="ie-viewbar">
            <button className="ie-btn icon sm ghost" onClick={() => zoomAtCenter(1 / ZOOM_STEP)} title="Zoom out (−)" type="button">
              −
            </button>
            <span className="ie-zoom">{zoomPct}%</span>
            <button className="ie-btn icon sm ghost" onClick={() => zoomAtCenter(ZOOM_STEP)} title="Zoom in (+)" type="button">
              +
            </button>
            <div className="ie-divider" style={{ height: "18px" }} />
            <button className="ie-btn sm ghost" onClick={fitToView} title="Fit to view (0)" type="button">
              Fit
            </button>
            <button className="ie-btn sm ghost" onClick={actualSize} title="Actual size (1)" type="button">
              100%
            </button>
          </div>
        ) : null}
      </main>

      {working ? (
        <aside className="ie-inspector" aria-label="Properties">
          <div className="ie-insp-head">
            <div className="ie-insp-icon">{EDITOR_TOOL_ICONS[tool]}</div>
            <div>
              <div className="ie-insp-title">{activeMeta.label}</div>
              <div className="ie-insp-desc">{activeMeta.desc}</div>
            </div>
          </div>
          {renderInspectorBody()}
        </aside>
      ) : null}

      {working ? renderLayers() : null}

      <footer className="ie-statusbar">
        <span className="ie-status-dot" />
        <span>{activeMeta.label}</span>
        <span className="ie-mono">·&nbsp; {zoomPct}%</span>
        {working ? (
          <span className="ie-mono">
            ·&nbsp; {working.width} × {working.height}
          </span>
        ) : null}
        <div className="ie-spacer" />
        <span className="ie-mono">{layerCount} layers</span>
        <span>·</span>
        <span>RGB / 8-bit</span>
        <span>·</span>
        <span>sRGB</span>
      </footer>
      {pickerOpen ? (
        <DatasetAddDialog
          assets={assets ?? []}
          characters={characters ?? []}
          confirmLabel="Open"
          eyebrow="Open"
          fileAccept="image/*"
          fileHint="Drag an image here, or"
          multiple={false}
          onAdd={async (ids) => {
            setPickerOpen(false);
            if (ids[0] && (await confirmDiscardEdits())) openAsset(ids[0]);
          }}
          onClose={() => setPickerOpen(false)}
          onImport={async (files) => {
            const file = files?.[0];
            setPickerOpen(false);
            if (file && (await confirmDiscardEdits())) openFile(file);
          }}
          title="Open image"
        />
      ) : null}

      {refPickerOpen ? (
        <DatasetAddDialog
          assets={assets ?? []}
          characters={characters ?? []}
          confirmLabel="Add"
          eyebrow="Reference"
          fileAccept="image/*"
          fileHint="Drag a reference image here, or"
          // Hide images already attached as references so the library tab only offers new picks.
          memberIds={refAssetIds}
          onAdd={(ids) => {
            setRefPickerOpen(false);
            setRefAssetIds((prev) =>
              Array.from(new Set([...prev, ...ids])).slice(0, MAX_EDIT_REFERENCES - 1),
            );
          }}
          onClose={() => setRefPickerOpen(false)}
          onImport={async (files) => {
            // Upload dropped images into the project, then attach them as references (sc-6107).
            setRefPickerOpen(false);
            const imported = await Promise.all(
              Array.from(files ?? []).map((file) => importAsset(file).catch(() => null)),
            );
            const ids = imported.filter(Boolean).map((asset) => asset.id);
            if (ids.length) {
              setRefAssetIds((prev) =>
                Array.from(new Set([...prev, ...ids])).slice(0, MAX_EDIT_REFERENCES - 1),
              );
            }
          }}
          title="Add reference image"
        />
      ) : null}

      {/* Portaled to document.body: the backdrop is `position: fixed`, which only
          anchors to the viewport when no ancestor establishes a containing block.
          Rendering it inline under the editor section left it vulnerable to being
          trapped inside a transformed/filtered ancestor (see Modal.jsx). */}
      {newLayoutOpen
        ? createPortal(
        <div
          className="image-editor-modal-backdrop"
          onClick={() => setNewLayoutOpen(false)}
          role="presentation"
        >
          <div
            aria-label="New blank layout"
            className="image-editor-modal"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
          >
            <h3 className="image-editor-modal-title">New blank layout</h3>
            <div className="image-editor-modal-field">
              <span>Aspect</span>
              <div className="image-editor-ratios" role="group" aria-label="Layout aspect">
                {EDIT_OUTPUT_ASPECTS.filter((aspect) => aspect.key !== "match").map((aspect) => (
                  <button
                    className={layoutAspect === aspect.key ? "active" : ""}
                    key={aspect.key}
                    onClick={() => setLayoutAspect(aspect.key)}
                    type="button"
                  >
                    {aspect.label}
                  </button>
                ))}
              </div>
            </div>
            <label className="image-editor-modal-field">
              <span>Size (long side)</span>
              <select onChange={(event) => setLayoutSize(Number(event.target.value))} value={layoutSize}>
                {BLANK_CANVAS_SIZES.map((size) => (
                  <option key={size} value={size}>
                    {size}px
                  </option>
                ))}
              </select>
            </label>
            <p className="image-editor-modal-dims">
              {layoutDims.width} × {layoutDims.height}px
            </p>
            <div className="image-editor-modal-actions">
              <button onClick={() => setNewLayoutOpen(false)} type="button">
                Cancel
              </button>
              <button className="primary" onClick={createBlankLayout} type="button">
                Create
              </button>
            </div>
          </div>
        </div>,
            document.body,
          )
        : null}
    </section>
  );
}
