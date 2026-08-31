import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Stage, Layer, Group, Image as KonvaImage, Line, Rect, Transformer } from "react-konva";
import { apiFetch, inspectWorkflowFile, isAbortError } from "../api.js";
import { terminalStatuses } from "../jobTypes.js";
import { useAppContext } from "../context/AppContext.js";
import { useScreenActive } from "../context/ScreenActiveContext.js";
import { appConfirm } from "../appConfirm.jsx";
import { isDesktop, tauriInvoke } from "../runtime.js";
import { DEFAULT_MAC_CAPABILITIES, macFeatureBlock } from "../macGating.js";
import { assetUrl, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import {
  SOURCE_WORKFLOW_ABSENT,
  SOURCE_WORKFLOW_PENDING,
  SOURCE_WORKFLOW_PRESENT,
  SOURCE_WORKFLOW_UNKNOWN,
  describeOpenedImage,
  workflowStateFromInspect,
} from "../editorSourceWorkflow.js";
import { SAVE_WITHOUT_WORKFLOW_LABEL } from "../workflowEmbed.js";
import { DatasetAddDialog } from "../components/DatasetAddDialog.jsx";
import { FitModeControl, effectiveFitMode } from "../components/FitModeControl.jsx";
import { useLoraSelection } from "../components/LoraPickerField.jsx";
import { findModelEditLora, loraIsInstalled } from "../presetUtils.js";
import { guidanceDefaultFromModel } from "../samplerOptions.js";
import {
  BLEND_MODES,
  DEFAULT_BLEND_MODE,
  activeLayerOf,
  addLayer,
  compositeLayersToCanvas,
  createLayer,
  drawLayerIntoCrop,
  duplicateLayer,
  identityTransform,
  layerById,
  moveLayer,
  removeLayer,
  replaceLayerBitmap,
  replaceLayerWithCroppedBitmap,
  sameLayerStack,
  setActiveLayer,
  setLayerProps,
  singleLayerWorking,
  snapshotLayers,
} from "../imageLayers.js";
import { CurveEditor } from "../components/CurveEditor.jsx";
import { StudioUpdateBadge, StudioUpdateNotice, updateOptionLabel } from "../components/StudioUpdateNotice.jsx";
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
import { maskAlphaFromRgba } from "../maskRefine.js";
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
import {
  applyColorKeyToRgba,
  applyMaskSelectionToRgba,
  documentPointToLayerPoint,
} from "./imageEditor/colorKeyMath.js";
import {
  ImageEditorToolPanel,
  useStableImageEditorToolPanelScope,
} from "./imageEditor/ImageEditorToolPanel.jsx";
import { EditorLoraPanel } from "./imageEditor/EditorLoraPanel.jsx";

const UPSCALE_ENGINE_DESC = {
  "real-esrgan": "Fast, faithful general-purpose upscaler. Great default.",
  seedvr2: "Detail-restoring diffusion upscaler for degraded sources.",
  "aura-sr": "Sharpest output, best for print. Slower.",
};

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
  applyColorKeyToRgba,
  documentPointToLayerPoint,
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
export const EDITOR_TOOL_ORDER = ["move", "transform", "crop", "upscale", "detail", "color", "cutout", "edit", "boxes"];
export const EDITOR_TOOL_META = {
  move: { label: "Move", desc: "Pan and inspect the canvas" },
  transform: { label: "Transform", desc: "Move, scale & rotate the layer" },
  crop: { label: "Crop", desc: "Trim to a size or ratio" },
  upscale: { label: "Upscale", desc: "Increase resolution with AI" },
  detail: { label: "Detail", desc: "Refine texture with tile ControlNet" },
  color: { label: "Color grade", desc: "Adjust tone, levels & curves" },
  cutout: { label: "Cutout", desc: "Remove a color-keyed background" },
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
  cutout: strokeSvg({ key: "cutout" }, [
    <path key="a" d="M4 5h16v14H4z" />, <path key="b" d="M7 15l3-4 2 2 2-3 3 5" />,
    <circle key="c" cx="9" cy="9" r="1.2" />,
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

// The browser CLOSE/REFRESH (beforeunload) warning shown while there is unsaved work, or
// null when a real unload would lose nothing (sc-2434 / sc-8850). Under keep-alive the
// editor stays mounted across in-app navigation, so a plain nav is non-destructive and gets
// NO prompt (see leaveGuardArming); only a genuine browser unload drops the in-memory state,
// which is what this wording addresses. Unsaved edits win the message. It also fires while an
// AI op is in flight — starting one does NOT set `dirty` (its result only lands on success) —
// so an unload mid-op is still flagged (its result would be lost).
export function leaveGuardMessage({ dirty, aiOpPending }) {
  if (dirty) return "You have unsaved edits in the Image Editor that will be lost if you close or reload.";
  if (aiOpPending) return "An image edit is still running; its result will be lost if you close or reload.";
  return null;
}

// Decide which leave-guard to arm (sc-11959 / sc-11968). Under selective keep-alive an in-app
// navigation away does NOT unmount the editor — unsaved edits, undo history, and an in-flight
// AI op all survive on the still-resident editor and the op's result lands back on return — so
// a plain nav is NON-DESTRUCTIVE and the in-app nav guard is permanently disarmed (`inApp:
// false`); prompting there would falsely warn users of loss that won't happen, undermining the
// keep-alive promise. Only a real browser close/refresh drops the in-memory state, so the
// beforeunload guard still arms whenever there are unsaved edits or an in-flight op — EVEN when
// the editor is backgrounded under keep-alive (an app close/refresh would otherwise silently
// discard a backgrounded editor's unsaved edits, a safeguard that predates keep-alive).
export function leaveGuardArming({ dirty, aiOpPending }) {
  const message = leaveGuardMessage({ dirty, aiOpPending });
  return {
    message,
    beforeUnload: Boolean(message),
    inApp: false,
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

export async function consumeEditorLaunch({ launch, confirmDiscard, openAsset, clearLaunch }) {
  if (!launch?.assetId || !(await confirmDiscard())) return false;
  await openAsset(launch.assetId);
  clearLaunch?.();
  return true;
}

export function createEditorLaunchGuard() {
  let activeId = null;
  return {
    async consume({ launch, isCurrent, confirmDiscard, openAsset, clearLaunch }) {
      if (!launch?.assetId || activeId === launch.id) return false;
      const launchId = launch.id;
      activeId = launchId;
      try {
        if (!(await confirmDiscard()) || activeId !== launchId || !isCurrent(launchId)) return false;
        await openAsset(launch.assetId);
        if (activeId !== launchId || !isCurrent(launchId)) return false;
        clearLaunch?.();
        return true;
      } finally {
        if (activeId === launchId) activeId = null;
      }
    },
  };
}

export function InlineLayerName({ layer, onRename }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(layer.name);

  if (!editing) {
    return (
      <button
        className="ie-layer-name"
        onClick={(event) => {
          event.stopPropagation();
          setDraft(layer.name);
          setEditing(true);
        }}
        title="Rename layer"
        type="button"
      >
        {layer.name}
      </button>
    );
  }

  return (
    <input
      aria-label={`Rename ${layer.name}`}
      autoFocus
      className="ie-input ie-layer-name-input"
      onBlur={() => {
        const name = draft.trim();
        if (name) onRename(layer.id, name);
        setEditing(false);
      }}
      onChange={(event) => setDraft(event.target.value)}
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => {
        if (event.key === "Enter") event.currentTarget.blur();
        if (event.key === "Escape") setEditing(false);
      }}
      value={draft}
    />
  );
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

// Import every selected reference independently so one bad file does not discard
// the successful imports. importAsset normally reports request errors itself; the
// dialog still needs an exact failure count instead of silently filtering them.
export async function importEditorReferenceFiles(files, importAsset) {
  const results = await Promise.allSettled(
    Array.from(files ?? []).map((file) => importAsset(file, { throwOnError: true })),
  );
  const assets = [];
  let failureCount = 0;
  for (const result of results) {
    if (result.status === "fulfilled" && result.value?.id) {
      assets.push(result.value);
    } else {
      failureCount += 1;
    }
  }
  return { assets, failureCount };
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
// A conservative cross-WebKit ceiling. WebKitGTK builds and GPU drivers vary in
// their exact canvas allocation limit; keeping both the longest side and total
// area bounded prevents the silent transparent-canvas failure seen above these
// values while retaining the largest normal SceneWorks generation size.
export const EDITOR_CANVAS_MAX_SIDE = 4096;
export const EDITOR_CANVAS_MAX_AREA =
  EDITOR_CANVAS_MAX_SIDE * EDITOR_CANVAS_MAX_SIDE;

export function boundedEditorCanvasDimensions(width, height) {
  const sourceWidth = Math.max(1, Math.round(Number(width) || 0));
  const sourceHeight = Math.max(1, Math.round(Number(height) || 0));
  const sideScale = Math.min(
    1,
    EDITOR_CANVAS_MAX_SIDE / sourceWidth,
    EDITOR_CANVAS_MAX_SIDE / sourceHeight,
  );
  const areaScale = Math.min(
    1,
    Math.sqrt(EDITOR_CANVAS_MAX_AREA / (sourceWidth * sourceHeight)),
  );
  const scale = Math.min(sideScale, areaScale);
  return {
    width: Math.max(1, Math.floor(sourceWidth * scale)),
    height: Math.max(1, Math.floor(sourceHeight * scale)),
    scaled: scale < 1,
    sourceWidth,
    sourceHeight,
  };
}

export async function exportEditorFile(
  file,
  {
    desktop = isDesktop,
    invoke = tauriInvoke,
    documentRef = globalThis.document,
    urlApi = globalThis.URL,
  } = {},
) {
  if (desktop) {
    return invoke("save_image_export", {
      imageBytes: Array.from(new Uint8Array(await file.arrayBuffer())),
      suggestedFilename: file.name,
    });
  }
  const url = urlApi.createObjectURL(file);
  const anchor = documentRef.createElement("a");
  anchor.href = url;
  anchor.download = file.name;
  documentRef.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  urlApi.revokeObjectURL(url);
  return null;
}

// ── What Download will actually contain (sc-15954, epic 15945) ───────────────
//
// sc-15948 puts the sanitized recipe INSIDE every generated PNG. Every other way a file leaves
// this app hands those bytes through unchanged, so the chunk survives for free. The editor was
// the exception: `Download` always went through `canvas.toBlob`, which re-encodes the composite
// and keeps nothing but pixels — so a user who opened a generated image, changed nothing, and
// downloaded it got a file with the recipe silently gone.
//
// The decision recorded on sc-15954, in one sentence:
//
//   An untouched PNG downloads the file you opened, byte for byte. A changed document downloads a
//   fresh PNG of what you see, with no recipe in it — and the header says which one you will get.
//
// # Why no `editedAfterGeneration` flag on the envelope
//
// The story's leaning was to re-embed the original envelope marked as provenance. Three reasons
// not to, in order of weight.
//
// 1. The envelope's extension rule is "adding a field does not need a version bump: an older
//    reader drops what it does not recognize" (docs/workflow-share-envelope.md). That rule holds
//    for fields whose absence is benign, and inverts for this one: an older build drops the flag
//    and then presents the envelope as a recipe that reproduces the image — the exact
//    misstatement this epic exists to stop, shipped by construction. Making it safe needs a
//    `schemaVersion` bump (which stops every older build reading every shared file, edited or
//    not) or a new marker kind that older builds refuse as `UnsupportedKind`. Either is a
//    contract change of the same size as the video lane, for a provenance note.
// 2. "The recipe this came from" is not well defined here. The document composites N layers from
//    N sources while `working.source` names one; an AI op replaces the base layer with the output
//    of a DIFFERENT job with its own model, prompt and seed; a crop changes the content and the
//    geometry.
// 3. It would be a new prompt egress, not a preserved one. The prompt is in the file as authored;
//    attaching it to a hand-edited composite the user believes is a fresh rasterization is a
//    surprise in the privacy direction too.
//
// # Why the geometry in the last AC cannot lie
//
// Structurally, not by a check. The only branch that ships an envelope ships the ORIGINAL FILE,
// so its `width`/`height` and its pixels are the same file's by definition. The branch that can
// be a different size — `boundedEditorCanvasDimensions` downscaling on load, a crop, an outpaint
// — is the branch that writes no envelope at all.
//
// # Why the untouched test is blob identity and not `dirty`
//
// `dirty` is not a statement about the bitmap: `runSave` clears it on a thoroughly edited
// document, so "not dirty" would let an edited composite take the passthrough branch. `edits` is
// not one either — layer transforms, opacity and blend changes never append to it. The one thing
// that IS a statement about the bitmap is reference identity between the single layer's blob and
// the blob the layer stack was installed from: every mutation the editor has (crop, colour grade,
// AI write-back, added layer, transform) replaces or supplements that object. Undo restores the
// same blob by reference (`snapshotLayers` shares it), so undoing back to the opened image
// correctly re-enables the passthrough.
//
// # Who decides whether the source carries a recipe
//
// `POST /api/v1/workflows/inspect`, which runs the one reader
// (`sceneworks_core::workflow_png::read_workflow_chunk`). Not a chunk walker in this app — see the
// header of `editorSourceWorkflow.js` for the five files a second implementation got wrong. The
// consequence for the plan is that the verdict is one of FOUR states and arrives late, so both
// `pending` and `unknown` have to survive all the way to the pill rather than being flattened into
// "no recipe" on the way.

/// The desktop shell's export ceiling, mirrored from `MAX_EXPORT_BYTES` in
/// `apps/desktop/src/settings.rs` and pinned to it by
/// `the_desktop_export_ceiling_matches_the_editors_mirror_of_it`.
///
/// It has to be known HERE rather than discovered from the error, because the byte-exact
/// passthrough is what can exceed it: before sc-15954 every editor download was a bounded canvas
/// raster and this limit was unreachable, and afterwards an untouched 300 MB source would hand the
/// shell 300 MB and get a hard "exceeds the 256 MB desktop limit" on a file that used to download
/// fine. So the plan falls back to the raster for those — the pre-sc-15954 behaviour, which still
/// works — and the pill says so BEFORE the click rather than the shell erroring after it. The
/// browser path has no such ceiling (it writes through an object URL) and keeps the passthrough.
export const MAX_DESKTOP_EXPORT_BYTES = 256 * 1024 * 1024;

/// Whether a layer sits 1:1 over the document with nothing done to it.
function layerIsUntouched(layer, working) {
  const transform = layer.transform ?? identityTransform();
  return (
    layer.visible !== false &&
    layer.opacity === 1 &&
    (layer.blendMode || DEFAULT_BLEND_MODE) === DEFAULT_BLEND_MODE &&
    transform.x === 0 &&
    transform.y === 0 &&
    transform.scaleX === 1 &&
    transform.scaleY === 1 &&
    transform.rotation === 0 &&
    working.width === (layer.image?.naturalWidth ?? -1) &&
    working.height === (layer.image?.naturalHeight ?? -1)
  );
}

/// Why an untouched document is being rasterized anyway. Null when it is not.
export const RASTER_REASON_EDITED = null;
export const RASTER_REASON_DESKTOP_CAP = "desktop-export-cap";

/// What `Download` will produce for this document. Pure, so the claim the header prints and the
/// bytes the export writes are decided in ONE place and cannot drift.
///
/// * `mode: "original"` — hand the opened file through untouched.
/// * `mode: "raster"` — flatten the layer stack to a new PNG, which carries no recipe ever.
///   `reason` says why, since one of the two reasons is not "you edited it".
///
/// `sourceWorkflow` is the reader's verdict about the OPENED file, one of the four
/// `SOURCE_WORKFLOW_*` states. It is a parameter and not a field on `working` because it resolves
/// asynchronously and must not be baked into an undo snapshot: the pill, the export branch and the
/// banner all read this one value, and `pending` / `unknown` reach all three intact.
export function editorDownloadPlan(
  working,
  sourceWorkflow = SOURCE_WORKFLOW_UNKNOWN,
  { desktop = isDesktop } = {},
) {
  const original = working?.source?.originalExport ?? null;
  // With no `originalExport` the source is not a PNG this editor can hand back — a JPEG, a blank
  // canvas, an AI result. None of those can carry an envelope, so the question is ANSWERED rather
  // than open, and `absent` is the honest state instead of `unknown`.
  const workflow = original ? sourceWorkflow : SOURCE_WORKFLOW_ABSENT;
  const present = workflow === SOURCE_WORKFLOW_PRESENT;
  const layers = working?.layers ?? [];
  const untouched =
    Boolean(original) &&
    layers.length === 1 &&
    layers[0].blob === original.installedBlob &&
    layerIsUntouched(layers[0], working);
  const overDesktopCap =
    untouched && desktop && Number(original.blob?.size) > MAX_DESKTOP_EXPORT_BYTES;
  if (untouched && !overDesktopCap) {
    return {
      mode: "original",
      reason: RASTER_REASON_EDITED,
      filename: original.filename,
      workflow,
      carriesWorkflow: present,
      hadWorkflow: present,
    };
  }
  return {
    mode: "raster",
    reason: overDesktopCap ? RASTER_REASON_DESKTOP_CAP : RASTER_REASON_EDITED,
    filename: editedFilename(working?.source),
    workflow,
    carriesWorkflow: false,
    hadWorkflow: present,
  };
}

/// The sentence about RESOLUTION, for a source the editor resampled on load — or "" when it did
/// not resample.
///
/// This exists because sc-15954 made the old banner false. `boundedEditorCanvasDimensions`
/// downscales anything over `EDITOR_CANVAS_MAX_SIDE`, and the banner used to say the scaling was
/// "for reliable WebKit editing and export" — true until the untouched branch started exporting
/// the ORIGINAL file at its original size. One sentence, rendered in both places that talk about
/// it (the canvas banner and the pill beside Download), so the two cannot say different things.
export function downloadSizeSentence(plan, downscaled) {
  if (!plan || !downscaled?.scaled) return "";
  const source = `${downscaled.sourceWidth} × ${downscaled.sourceHeight}`;
  const working = `${downscaled.width} × ${downscaled.height}`;
  if (plan.mode === "original") {
    return `Download saves the original file at ${source}, not the ${working} working copy on the canvas.`;
  }
  if (plan.reason === RASTER_REASON_DESKTOP_CAP) {
    return `Download writes a new PNG at the ${working} working size, not the original ${source}.`;
  }
  return `This image has been edited, so Download writes a new PNG at the ${working} working size rather than the original ${source}.`;
}

/// The desktop-ceiling sentence, or "".
function desktopCapSentence(plan) {
  if (plan?.reason !== RASTER_REASON_DESKTOP_CAP) return "";
  return (
    `This file is larger than the ${Math.round(MAX_DESKTOP_EXPORT_BYTES / (1024 * 1024))} MB ` +
    "the desktop save dialog accepts, so Download writes a new PNG of the working copy instead of " +
    "handing the original file through."
  );
}

function joinSentences(...parts) {
  return parts.filter(Boolean).join(" ");
}

/// The line beside the Download button, or null when there is genuinely nothing to say — an
/// ordinary-sized image the reader confirmed carries no recipe, which is most images in the world.
///
/// Every other combination gets a sentence, including the two that a first cut left silent:
///
/// * **`pending` / `unknown`** render rather than reading as "no recipe". A verdict that has not
///   arrived, or one the reader could not produce, is a different fact from a verdict of "no", and
///   collapsing the three is what let an over-ceiling export ship a recipe with no pill at all.
/// * **A resampled source with no recipe** renders too, because sc-15954 changed which RESOLUTION
///   the file downloads at and `hadWorkflow` has nothing to do with that.
export function editorDownloadNote(plan, { downscaled = null } = {}) {
  if (!plan) return null;
  const size = downloadSizeSentence(plan, downscaled);
  const cap = desktopCapSentence(plan);
  if (plan.workflow === SOURCE_WORKFLOW_PRESENT) {
    if (plan.mode === "original") {
      return {
        tone: "included",
        label: "Recipe included",
        detail: joinSentences(
          "Nothing has been changed, so Download saves the file you opened byte for byte — " +
            "including the SceneWorks recipe embedded in it.",
          `To send a copy without the recipe, use “${SAVE_WITHOUT_WORKFLOW_LABEL}” on this image ` +
            "in the Library — saving from the editor rasterizes, so it drops the recipe along " +
            "with everything else the file carries.",
          size,
        ),
      };
    }
    return {
      tone: "dropped",
      label: "Recipe not carried",
      detail: joinSentences(
        cap ||
          "This image has been edited, so Download writes a new PNG of what you see. The " +
            "SceneWorks recipe from the image you opened is not in it — it describes a run that " +
            "did not produce these pixels.",
        cap && "The SceneWorks recipe embedded in the original is not in it.",
        size,
      ),
    };
  }
  if (plan.workflow === SOURCE_WORKFLOW_PENDING) {
    return {
      tone: "pending",
      label: "Checking for a recipe…",
      detail: joinSentences(
        "SceneWorks is reading the file you opened to find out whether it carries an embedded " +
          "recipe. Until that answers, what Download will contain is not yet known.",
        size,
      ),
    };
  }
  if (plan.workflow === SOURCE_WORKFLOW_UNKNOWN) {
    return {
      tone: "unknown",
      label: "Recipe unknown",
      detail: joinSentences(
        "SceneWorks could not read the file you opened, so whether it carries an embedded recipe " +
          "is unknown — not that it has none.",
        plan.mode === "original"
          ? "Nothing has been changed, so Download saves the file byte for byte and anything in " +
              "it travels."
          : joinSentences(
              cap ||
                "This image has been edited, so Download writes a new PNG of what you see and " +
                  "anything the original carried is not in it.",
              cap && "Anything the original carried is not in it.",
            ),
        size,
      ),
    };
  }
  // `absent`: the reader walked the file and found nothing, so there is no recipe to speak about.
  // What is left is what sc-15954 changed about the FILE ITSELF — which resolution it downloads
  // at, and whether it is the original at all. Both are silent otherwise.
  if (!size && !cap) return null;
  return {
    tone: "size",
    label: cap
      ? "Not the original file"
      : plan.mode === "original"
        ? "Original size"
        : "Working size",
    detail: joinSentences(cap, size),
  };
}

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

function colorKeyCanvasForImage(image, seed, options) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const cutout = applyColorKeyToRgba(imageData.data, canvas.width, canvas.height, { ...options, ...seed });
  imageData.data.set(cutout.rgba);
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function maskCutoutCanvasForImage(image, maskCanvas, keepSelected) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const mask = maskCanvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
  imageData.data.set(applyMaskSelectionToRgba(imageData.data, maskAlphaFromRgba(mask.data), keepSelected));
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function canvasToPngBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not encode the cutout."))), "image/png");
  });
}

// Decode an editor source and, only when it exceeds the conservative WebKit
// canvas ceiling, rasterize it directly into a bounded surface before any Konva
// or export canvas is allocated. The caller owns the returned object URL.
async function blobToEditorImage(blob) {
  const decoded = await blobToImage(blob);
  const dimensions = boundedEditorCanvasDimensions(
    decoded.image.naturalWidth,
    decoded.image.naturalHeight,
  );
  if (!dimensions.scaled) {
    return { ...decoded, blob, downscaled: null };
  }

  const canvas = document.createElement("canvas");
  canvas.width = dimensions.width;
  canvas.height = dimensions.height;
  let resizedBlob;
  try {
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Could not create a 2D editor canvas.");
    }
    context.drawImage(
      decoded.image,
      0,
      0,
      dimensions.width,
      dimensions.height,
    );
    resizedBlob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/png"),
    );
  } finally {
    URL.revokeObjectURL(decoded.objectUrl);
  }
  if (!resizedBlob) {
    throw new Error("Could not prepare this large image for the editor.");
  }
  const resized = await blobToImage(resizedBlob);
  return {
    ...resized,
    blob: resizedBlob,
    downscaled: dimensions,
  };
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
  // Selective keep-alive (sc-11959): this editor stays mounted (merely hidden) once visited,
  // so it must consult this flag to avoid acting while it is the backgrounded view — the
  // window keydown handler below is gated on it (sc-13589).
  const screenActive = useScreenActive();
  // Mac UI gating (sc-3486): the upscale tool itself runs in-process on Rust (Real-ESRGAN,
  // sc-3489), so it is available on a gated Mac — this block is a defensive guard that stays
  // null. The second engine (AuraSR) is dropped on Mac (sc-3668) and gated per-engine below.
  const macUpscaleBlock = macFeatureBlock(macCapabilities, "imageUpscale");
  // Smart-select runs native SAM3 on MLX (Mac) and Candle (Windows/Linux). Gate it on the
  // platform-intrinsic `imageSegment` capability, independent of the Mac gating-rollout switch.
  // When false or still loading, the mask tool shows only the hand brush (graceful degradation).
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
  const launchAttemptRef = useRef(null);
  const launchGuardRef = useRef(null);
  if (!launchGuardRef.current) launchGuardRef.current = createEditorLaunchGuard();
  const editorLaunchRef = useRef(editorLaunch);
  editorLaunchRef.current = editorLaunch;
  const editorMountedRef = useRef(true);
  useEffect(() => {
    editorMountedRef.current = true;
    return () => {
      editorMountedRef.current = false;
    };
  }, []);

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
  // Color-key cutout (sc-22462): selection/preview remain ephemeral until Apply,
  // then the active layer receives one alpha-only bitmap replacement.
  const [colorKeySeed, setColorKeySeed] = useState(null);
  const [colorKeyTolerance, setColorKeyTolerance] = useState(12);
  const [colorKeySoftness, setColorKeySoftness] = useState(8);
  const [colorKeyGlobal, setColorKeyGlobal] = useState(false);
  const [cutoutKeepSelected, setCutoutKeepSelected] = useState(true);
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
  const selectedDetailModel = detailModels.find((model) => model.id === detailModel) ?? null;
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
  const requestEditLoraDownload = useCallback(async () => {
    if (!editLora) return;
    setEditLoraDownloadRequested(true);
    try {
      const job = await createLoraDownloadJob?.(editLora);
      if (!job) setEditLoraDownloadRequested(false);
    } catch {
      setEditLoraDownloadRequested(false);
    }
  }, [editLora, createLoraDownloadJob]);
  useEffect(() => {
    if (!editLora?.updateAvailable) setEditLoraDownloadRequested(false);
  }, [editLora?.updateAvailable]);
  // The manual LoRA picker hides the managed edit LoRA (it's applied automatically), so it can't be
  // double-shown or accidentally toggled — mirrors the Studio's pickerCompatibleLoras.
  const pickerCompatibleLoras = useMemo(
    () => managedEditLoraId
      ? editLoraSelection.compatibleLoras.filter((lora) => lora.id !== managedEditLoraId)
      : editLoraSelection.compatibleLoras,
    [editLoraSelection.compatibleLoras, managedEditLoraId],
  );
  const [showIncompatibleEditLoras, setShowIncompatibleEditLoras] = useState(false);
  const editorPickerLoras = pickerCompatibleLoras;
  const selectedEditLoras = useMemo(
    () => editLoraSelection.selectedLoraIds
      .map((id) => pickerCompatibleLoras.find((lora) => lora.id === id))
      .filter(Boolean),
    [editLoraSelection.selectedLoraIds, pickerCompatibleLoras],
  );
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
  const mountedRef = useRef(false);
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

  // Preview is an offscreen canvas, never a layer mutation or a history entry.
  // The stage swaps it in only for the selected active layer, retaining that
  // layer's normal visibility, opacity, blend mode, and transform.
  const colorKeyPreview = useMemo(() => {
    const layer = activeLayerOf(working);
    if (!layer?.image || !colorKeySeed) return null;
    try {
      return {
        layerId: layer.id,
        image: colorKeyCanvasForImage(layer.image, colorKeySeed, {
          tolerance: colorKeyTolerance,
          softness: colorKeySoftness,
          global: colorKeyGlobal,
        }),
      };
    } catch {
      return null;
    }
  }, [working, colorKeySeed, colorKeyTolerance, colorKeySoftness, colorKeyGlobal]);

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
    smartSelectSupported,
    aiOp,
    activeProject,
    requestedGpu,
    runAiOp: runAiOpBridge,
    stagePointToImage: stagePointToActiveLayerImage,
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
    rasterizeMaskToCanvas,
    refineMask,
    resetMaskState,
  } = maskTool;

  // SAM3 cutout preview is an offscreen alpha composition, just like color key:
  // refinement remains editable and no document/history mutation occurs until Apply.
  const maskCutoutPreview = useMemo(() => {
    const layer = activeLayerOf(working);
    if (tool !== "cutout" || !maskMode || !layer?.image || (!maskHasContent(maskLines) && !maskBaseImage)) return null;
    try {
      return {
        layerId: layer.id,
        image: maskCutoutCanvasForImage(layer.image, rasterizeMaskToCanvas(), cutoutKeepSelected),
      };
    } catch {
      return null;
    }
  }, [working, tool, maskMode, maskLines, maskBaseImage, rasterizeMaskToCanvas, cutoutKeepSelected]);

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

  // Revoke every live layer's object URL when the editor unmounts and mark
  // in-flight terminal-result decodes as stale.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      revokeLayerUrls(workingRef.current?.layers);
    };
  }, []);

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
    setColorKeySeed(null);
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
    // Under selective keep-alive (sc-11959) this editor stays mounted and this window
    // listener stays subscribed even when another view is foregrounded, so gate every
    // shortcut on the active flag — a backgrounded editor must never undo/redo, delete a
    // box, switch tools, or toggle the reference in response to keys meant for the visible
    // screen (sc-13589). The handler is reassigned each render, so `screenActive` is fresh.
    if (!screenActive) return;
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
        const prepared = await blobToEditorImage(blob);
        // Keep the bytes we opened (sc-15954). The blob is retained by reference — no copy — and
        // `installedBlob` is the identity token `editorDownloadPlan` compares the layer stack
        // against. Note it is `prepared.blob`, not `blob`: a source over EDITOR_CANVAS_MAX_SIDE is
        // resampled on load, so the two differ exactly in the case the story calls the sharp edge,
        // and the passthrough branch must ship the ORIGINAL rather than the proxy the canvas is
        // showing. Whether those bytes carry a recipe is NOT decided here — `workflow` is a
        // starting state and the effect below asks the one reader.
        const opened = await describeOpenedImage(blob, source?.name);
        const preparedSource = {
          ...source,
          ...(prepared.downscaled ? { editorDownscaled: prepared.downscaled } : null),
          ...(opened ? { originalExport: { ...opened, installedBlob: prepared.blob } } : null),
        };
        installWorkingImage(
          prepared.image,
          prepared.objectUrl,
          prepared.blob,
          preparedSource,
        );
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
  // { id, assetId } in editorLaunch. A dirty editor confirms before replacement. Cancelling
  // leaves the launch pending; changing the dirty state re-arms it, while the attempt/in-flight
  // refs prevent ordinary re-renders or the accepted open's own state updates from duplicating
  // the prompt/load. Entering via the nav with no launch leaves editorLaunch null. We clear
  // only after the accepted open has finished handling the asset.
  useEffect(() => {
    if (!editorLaunch?.assetId) return;
    const attemptKey = `${editorLaunch.id}:${dirty}`;
    if (launchAttemptRef.current === attemptKey) return;
    launchAttemptRef.current = attemptKey;
    launchGuardRef.current.consume({
      launch: editorLaunch,
      isCurrent: (id) => editorMountedRef.current && editorLaunchRef.current?.id === id,
      confirmDiscard: confirmDiscardEdits,
      openAsset,
      clearLaunch: clearEditorLaunch,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editorLaunch?.id, dirty]);

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
    if (tool === "cutout") {
      setColorKeySeed(null);
      setTool("move");
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

  // Commit the preview's exact alpha calculation to the active layer. The
  // checkpoint is deliberately after encode/decode succeeds and immediately
  // before the one bitmap replacement, so a failed encode has no history entry
  // and Apply always creates exactly one undoable operation.
  const applyColorKey = useCallback(async () => {
    const work = workingRef.current;
    const layer = activeLayerOf(work);
    if (!layer?.image || !colorKeySeed) return;
    const sourceBlob = layer.blob;
    try {
      const canvas = colorKeyCanvasForImage(layer.image, colorKeySeed, {
        tolerance: colorKeyTolerance,
        softness: colorKeySoftness,
        global: colorKeyGlobal,
      });
      const blob = await canvasToPngBlob(canvas);
      const { image, objectUrl } = await blobToImage(blob);
      const current = layerById(workingRef.current, layer.id);
      if (!current || current.blob !== sourceBlob) {
        URL.revokeObjectURL(objectUrl);
        return;
      }
      checkpoint();
      replaceLayerImage(layer.id, image, objectUrl, blob);
      setEdits((prev) => [
        ...prev,
        { op: "colorKey", mode: colorKeyGlobal ? "global" : "connected", tolerance: colorKeyTolerance, softness: colorKeySoftness },
      ]);
      setDirty(true);
      setColorKeySeed(null);
      setTool("move");
    } catch (err) {
      setStatus({ loading: false, error: err.message || "Could not apply the color-key cutout." });
    }
  }, [colorKeySeed, colorKeyTolerance, colorKeySoftness, colorKeyGlobal, checkpoint, replaceLayerImage]);

  // Commit the currently refined SAM3 mask to the active layer only. The mask is
  // converted to an alpha multiplier, so keep/remove choices both preserve an
  // already-transparent source pixel and produce one undoable bitmap replacement.
  const applyMaskCutout = useCallback(async () => {
    const work = workingRef.current;
    const layer = activeLayerOf(work);
    if (!layer?.image || (!maskHasContent(maskLines) && !maskBaseImage)) return;
    const sourceBlob = layer.blob;
    try {
      const canvas = maskCutoutCanvasForImage(layer.image, rasterizeMaskToCanvas(), cutoutKeepSelected);
      const blob = await canvasToPngBlob(canvas);
      const { image, objectUrl } = await blobToImage(blob);
      const current = layerById(workingRef.current, layer.id);
      if (!current || current.blob !== sourceBlob) {
        URL.revokeObjectURL(objectUrl);
        return;
      }
      checkpoint();
      replaceLayerImage(layer.id, image, objectUrl, blob);
      setEdits((prev) => [...prev, { op: "sam3Cutout", mode: cutoutKeepSelected ? "keepSelected" : "removeSelected" }]);
      setDirty(true);
      clearMask();
      setTool("move");
    } catch (err) {
      setStatus({ loading: false, error: err.message || "Could not apply the SAM3 cutout." });
    }
  }, [maskLines, maskBaseImage, rasterizeMaskToCanvas, cutoutKeepSelected, checkpoint, replaceLayerImage, clearMask]);

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
    // Crop every layer in document space. Per-layer transforms are baked into the
    // new bitmap so the result matches the visible composition; layer compositing
    // metadata remains intact and the baked transform is reset below.
    let cropped;
    try {
      cropped = await Promise.all(
        working.layers.map(async (layer) => {
          const canvas = document.createElement("canvas");
          canvas.width = sw;
          canvas.height = sh;
          const ctx = canvas.getContext("2d");
          drawLayerIntoCrop(ctx, layer, { x: sx, y: sy, width: sw, height: sh }, straighten);
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
        return c ? replaceLayerWithCroppedBitmap(layer, c) : layer;
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
    colorKeyPointerDown(event);
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

  // SAM3 receives a scratch image of the active layer, not the composited
  // document. Convert the visible-stage drag through that layer's transform so
  // the worker box and the editable mask use the bitmap's own pixel coordinates.
  function stagePointToActiveLayerImage(event) {
    const point = stagePointToImage(event);
    const layer = activeLayerOf(workingRef.current);
    if (!point || !layer?.image) return null;
    const local = documentPointToLayerPoint(point, layer.transform);
    if (!local) return null;
    return {
      x: clamp(local.x, 0, layer.image.naturalWidth),
      y: clamp(local.y, 0, layer.image.naturalHeight),
    };
  }

  // The canvas reports document coordinates, while each layer can be translated,
  // rotated, or scaled. Invert the active layer transform before sampling its
  // source bitmap so the eyedropper and the eventual alpha write address the
  // same pixels.
  function colorKeyPointerDown(event) {
    // When cutout is in SAM3 mask mode the canvas gesture belongs to the brush/
    // selection box, not to the color-key eyedropper.
    if (tool !== "cutout" || maskMode) return;
    const layer = activeLayerOf(workingRef.current);
    const point = stagePointToImage(event);
    if (!layer?.image || !point) return;
    const local = documentPointToLayerPoint(point, layer.transform);
    if (
      !local ||
      local.x < 0 ||
      local.y < 0 ||
      local.x >= layer.image.naturalWidth ||
      local.y >= layer.image.naturalHeight
    ) return;
    setColorKeySeed({ x: Math.floor(local.x), y: Math.floor(local.y) });
  }

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
        const prepared = await blobToEditorImage(blob);
        if (!mountedRef.current) {
          URL.revokeObjectURL(prepared.objectUrl);
          return;
        }
        checkpoint();
        // Active-layer op (same-size edit / detail) → write the result back into the
        // target layer, preserving the rest of the stack; document op (upscale /
        // outpaint / box edit) → flatten the stack into one new base layer (sc-6119).
        if (writeBack === "activeLayer" && targetLayerId && layerById(workingRef.current, targetLayerId)) {
          replaceLayerImage(
            targetLayerId,
            prepared.image,
            prepared.objectUrl,
            prepared.blob,
          );
        } else {
          const preparedSource = prepared.downscaled
            ? { ...source, editorDownscaled: prepared.downscaled }
            : source;
          installWorkingImage(
            prepared.image,
            prepared.objectUrl,
            prepared.blob,
            preparedSource,
          );
        }
        if (edit) setEdits((prev) => [...prev, edit]);
        setDirty(true);
      } catch (err) {
        if (mountedRef.current) {
          setStatus({ loading: false, error: err.message || "The operation failed." });
        }
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

  // ── Does the opened file carry a recipe? Ask the one reader (sc-15954) ─────
  //
  // `POST /api/v1/workflows/inspect` runs `read_workflow_chunk`, the same function the import path
  // and the sc-15951 drop panel use. Nothing in this app parses PNG chunks.
  //
  // The verdict lives HERE and not on `working.source`, for two reasons. It resolves after the
  // document is installed, so writing it back through `setWorking` would either race the user's
  // first edit or bake a stale `pending` into every undo snapshot taken before it landed. And it
  // is keyed by the source blob, which undo restores BY REFERENCE — so undoing back to the opened
  // image reuses the answer instead of asking again.
  const [sourceWorkflow, setSourceWorkflow] = useState({ blob: null, state: SOURCE_WORKFLOW_UNKNOWN });
  const openedExport = working?.source?.originalExport ?? null;
  const openedBlob = openedExport?.blob ?? null;
  const openedStartState = openedExport?.workflow ?? SOURCE_WORKFLOW_UNKNOWN;
  const openedFilename = openedExport?.filename ?? null;
  const activeProjectId = activeProject?.id ?? null;

  useEffect(() => {
    // `describeOpenedImage` already decided whether asking is possible at all: a PNG past the
    // endpoint's own cap starts `unknown` and is never uploaded, because staging half a gigabyte
    // server-side to be handed a 413 is a slower way to learn nothing.
    if (!openedBlob || openedStartState !== SOURCE_WORKFLOW_PENDING) return undefined;
    let live = true;
    const controller = new AbortController();
    setSourceWorkflow({ blob: openedBlob, state: SOURCE_WORKFLOW_PENDING });
    (async () => {
      let state = SOURCE_WORKFLOW_UNKNOWN;
      try {
        const body = await inspectWorkflowFile(
          // Named so the multipart part carries a filename, exactly as the drop path's `File`
          // does. `projectId` only widens the LoRA/preset lookups the resolution report is built
          // against, and this caller reads `status` alone — but it is passed anyway so the server
          // does one kind of work for one kind of request.
          new File([openedBlob], openedFilename ?? "image.png", { type: "image/png" }),
          { projectId: activeProjectId, token, signal: controller.signal },
        );
        state = workflowStateFromInspect(body);
      } catch (err) {
        // Offline, a 413, a 500, a 422 for a file this build refuses to read — none of those is
        // "no recipe", and reporting them as one is the silent loss the AC forbids. `unknown` is
        // rendered, not swallowed.
        if (isAbortError(err)) return;
        state = SOURCE_WORKFLOW_UNKNOWN;
      }
      if (live) setSourceWorkflow({ blob: openedBlob, state });
    })();
    return () => {
      live = false;
      controller.abort();
    };
  }, [openedBlob, openedStartState, openedFilename, activeProjectId, token]);

  // The verdict for THIS document. Before the effect has run — and for any blob the store is not
  // holding an answer for — it falls back to the state the open stamped, which is `pending` for a
  // file on its way to the reader and `unknown` for one that will never go. Never `absent`.
  const sourceWorkflowState =
    sourceWorkflow.blob && sourceWorkflow.blob === openedBlob
      ? sourceWorkflow.state
      : openedStartState;

  // The one decision the pill, the banner and the export branch all read.
  const downloadPlan = useMemo(
    () => (working ? editorDownloadPlan(working, sourceWorkflowState) : null),
    [working, sourceWorkflowState],
  );
  const downloadNote = useMemo(
    () =>
      downloadPlan
        ? editorDownloadNote(downloadPlan, { downscaled: working?.source?.editorDownscaled ?? null })
        : null,
    [downloadPlan, working],
  );

  // Export straight to disk (no project involvement). An untouched PNG goes out as the file it
  // came in as — same bytes, same recipe chunk, same colour profile — and anything else is
  // flattened to a fresh PNG with no recipe in it (sc-15954). `downloadNote`, rendered beside the
  // button, says which of the two the click will produce — and the click reads the SAME plan
  // object the pill was rendered from, so the two cannot disagree about the file that lands.
  const runDownload = useCallback(async () => {
    if (!working || !downloadPlan) return;
    try {
      const file =
        downloadPlan.mode === "original"
          ? // The File wraps the SAME blob — no re-encode, so this is the one export path in the
            // editor whose bytes are the source's bytes.
            new File([working.source.originalExport.blob], downloadPlan.filename, {
              type: "image/png",
            })
          : await workingImageToFile(downloadPlan.filename);
      await exportEditorFile(file);
    } catch (err) {
      setStatus({ loading: false, error: `Could not export: ${err.message || err}` });
    }
  }, [working, downloadPlan, workingImageToFile]);

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

  // Warn before a browser CLOSE/REFRESH that would drop unsaved edits OR an in-flight AI op
  // (sc-2434 / sc-8850 / sc-11968). Under keep-alive an in-app navigation away does NOT
  // unmount this editor (App keeps it resident), so unsaved edits, undo history, and an
  // in-flight op all SURVIVE the nav and the op's result lands back on return — a plain nav
  // is non-destructive and shows NO prompt. Only a real unload drops the in-memory state, so
  // just the beforeunload guard arms here (even when backgrounded). Starting an AI op does
  // NOT set `dirty` — its result only lands on success — so the guard must also arm while
  // `aiOp` is non-null, else an unload mid-op silently loses the result. The intentional
  // lose-work prompts live elsewhere: the explicit Close/Discard button (closeDoc) and the
  // open-new / flatten confirms, all via the desktop-safe appConfirm dialog.
  const aiOpPending = Boolean(aiOp);
  useEffect(() => {
    const { beforeUnload } = leaveGuardArming({ dirty, aiOpPending });
    if (!beforeUnload) return undefined;
    const onBeforeUnload = (event) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
    };
  }, [dirty, aiOpPending]);

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
    else if (key === "cutout") setTool("cutout");
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
    cutout: maskMode
      ? (maskSubTool === "select" ? "Drag a box to select an object with SAM3" : "Paint to refine the SAM3 selection")
      : (colorKeySeed ? "Adjust the key, then apply the alpha cutout" : "Click the background color to preview a cutout"),
    edit: maskMode ? "Paint or box-select the region to edit" : "Describe the edit on the right",
    boxes: "Drag to draw a region, then describe it",
  }[tool];

  // Short engine blurbs for the upscale radio cards (design copy). Keyed by the
  // platform engine list (`availableUpscaleEngines`); unknown keys get no blurb.
  const setCropDim = useCallback((dim, raw) => {
    const value = Number(raw);
    if (!cropRect || !working || !Number.isFinite(value)) return;
    const rect = { ...cropRect, [dim]: value };
    const width = clamp(rect.width, MIN_CROP_PX, working.width);
    const height = clamp(rect.height, MIN_CROP_PX, working.height);
    setCropRect({
      width,
      height,
      x: clamp(rect.x, 0, working.width - width),
      y: clamp(rect.y, 0, working.height - height),
    });
  }, [cropRect, working]);

  const toolPanelScope = {
    BOX_PALETTE,
    COLOR_ADJUSTMENTS,
    CROP_RATIOS,
    CurveEditor,
    EDIT_OUTPUT_ASPECTS,
    EditorLoraPanel,
    FitModeControl,
    MAX_BOX_PALETTE,
    MAX_EDIT_REFERENCES,
    StudioUpdateBadge,
    StudioUpdateNotice,
    UPSCALE_ENGINE_DESC,
    activeGradeIsIdentity,
    activeLayerOf,
    actualSize,
    addPaletteColor,
    aiOp,
    applyColorGrade,
    applyColorKey,
    applyMaskCutout,
    applyCrop,
    assetUrl,
    availableUpscaleEngines,
    boxColor,
    boxMetadataGaps,
    boxes,
    canMask,
    cancelCrop,
    channelStroke,
    chooseBoxColor,
    chooseRatio,
    clearBoxes,
    clearMask,
    colorAdjust,
    colorChannel,
    colorKeyGlobal,
    colorKeySeed,
    colorKeySoftness,
    colorKeyTolerance,
    colorMode,
    composedPrompt,
    createLoraDownloadJob,
    createModelDownloadJob,
    cropRect,
    cutoutKeepSelected,
    curves,
    deleteBox,
    detailCnScale,
    detailModel,
    detailModels,
    detailStrength,
    editAspect,
    editFitMode,
    editGuidance,
    editLora,
    editLoraDownloadRequested,
    editLoraInstalled,
    editLoraRequiredMissing,
    editLoraSelection,
    editModel,
    editModels,
    editPrompt,
    editSeed,
    editorPickerLoras,
    effectiveFitMode,
    endTransformGesture,
    fitToView,
    flipActiveLayer,
    guidanceDefaultFromModel,
    histogramRef,
    identityTransform,
    imageAssets,
    layerCount,
    levels,
    maskActive,
    maskBaseImage,
    maskBrush,
    maskErase,
    maskHasContent,
    maskLines,
    maskMode,
    maskRefineRadius,
    maskSubTool,
    multiRefCapable,
    onTransformSlider,
    ratioKey,
    refAssetIds,
    refineMask,
    removePaletteColor,
    requestEditLoraDownload,
    requestTileControlNetDownload,
    resetActiveColorMode,
    resetActiveLayerTransform,
    resetAdjust,
    rotated,
    runBoxEdit,
    runDetail,
    runEdit,
    runUpscale,
    selectedBox,
    selectedBoxGaps,
    selectedBoxId,
    selectedDetailModel,
    selectedEditLoras,
    selectedEditModel,
    setActiveTransform,
    setAdjustValue,
    setColorChannel,
    setColorKeyGlobal,
    setColorKeySeed,
    setColorKeySoftness,
    setColorKeyTolerance,
    setColorMode,
    setCropDim,
    setCurves,
    setCutoutKeepSelected,
    setDetailCnScale,
    setDetailModel,
    setDetailStrength,
    setEditAspect,
    setEditFitMode,
    setEditGuidance,
    setEditModel,
    setEditPrompt,
    setEditSeed,
    setLevelsValue,
    setMaskBrush,
    setMaskErase,
    setMaskMode,
    setMaskRefineRadius,
    setMaskSubTool,
    setRefAssetIds,
    setRefPickerOpen,
    setSelectedBoxId,
    setShowIncompatibleEditLoras,
    setStraighten,
    setTool,
    setUpscaleEngine,
    setUpscaleFactor,
    setUpscaleSoftness,
    showIncompatibleEditLoras,
    smartSelectSupported,
    straighten,
    tileControlNet,
    tileControlNetDownloadRequested,
    tileControlNetReady,
    toggleRotate,
    updateBox,
    updateOptionLabel,
    upscaleEngine,
    upscaleEngineHasSoftness,
    upscaleFactor,
    upscaleFactorsForEngine,
    upscaleSoftness,
    working,
  };
  const stableToolPanelScope = useStableImageEditorToolPanelScope(toolPanelScope);
  const renderToolPanel = (key) => <ImageEditorToolPanel panelKey={key} scope={stableToolPanelScope} />;

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
                    <InlineLayerName layer={layer} onRename={renameLayer} />
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
            {/* Whether the download will carry the embedded recipe (sc-15954). Visible text, not
                a tooltip: "silent loss is not an acceptable outcome" is the AC, and a title
                attribute is silence on a touch device.

                Mounted UNCONDITIONALLY for the whole life of the document and emptied rather than
                unmounted. A live region inserted into the DOM already populated is not reliably
                announced by any screen reader — the region has to exist before its content
                changes for the change to be a change — and this one's content genuinely does
                change under the user: "Checking for a recipe…" becomes "Recipe included" when the
                reader answers, and "Recipe not carried" the moment they crop. */}
            <span
              className="ie-recipe-note"
              data-empty={downloadNote ? undefined : "true"}
              data-tone={downloadNote?.tone ?? "none"}
              role="status"
              title={downloadNote?.detail ?? undefined}
            >
              {downloadNote?.label ?? ""}
            </span>
            <button
              className="ie-btn sm"
              onClick={runDownload}
              title={downloadNote ? `Download a PNG to your computer — ${downloadNote.detail}` : "Download a PNG to your computer"}
              type="button"
            >
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
        {/* The canvas is a proxy for an over-ceiling source. The second sentence is not a
            flourish: before sc-15954 the downscale applied to the export too, and this banner said
            so ("…for reliable WebKit editing and export"). Now an untouched download hands back
            the ORIGINAL file at its original size, which makes the old sentence the only
            user-visible statement about the downscale and a false one. It is rendered from
            `downloadSizeSentence`, the same helper the pill's detail uses, so the canvas and the
            button cannot describe the same click differently — and it follows the PLAN, so the
            desktop-ceiling fallback (where an untouched download IS the working copy) reads
            correctly too. */}
        {working?.source?.editorDownscaled ? (
          <div
            className="ie-hint"
            role="status"
            style={{ top: "58px", borderColor: "color-mix(in srgb, var(--ie-warn) 55%, var(--ie-border))" }}
          >
            <span>
              Large source scaled from{" "}
              {working.source.editorDownscaled.sourceWidth} x{" "}
              {working.source.editorDownscaled.sourceHeight} to {working.width} x{" "}
              {working.height} for reliable WebKit editing.{" "}
              {downloadSizeSentence(downloadPlan, working.source.editorDownscaled)}
            </span>
          </div>
        ) : null}
        {working && stageSize.width > 0 && stageSize.height > 0 ? (
          <Stage
            draggable={tool !== "crop" && tool !== "boxes" && tool !== "transform" && tool !== "cutout" && !maskMode}
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
                    image={isActive && maskCutoutPreview?.layerId === layer.id
                      ? maskCutoutPreview.image
                      : (isActive && colorKeyPreview?.layerId === layer.id ? colorKeyPreview.image : layer.image)}
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
            {maskMode && (maskLines.length || maskOverlay) ? (
              // Isolated layer so the eraser's destination-out clears only the mask
              // overlay, never the image beneath it. The smart-select base (sc-3751)
              // renders first, with the brush strokes (and their erases) composited on top.
              // The group mirrors the active layer transform: the mask's coordinates
              // are the staged active bitmap's coordinates, never document coordinates.
              <Layer listening={false}>
                {(() => {
                  const active = activeLayerOf(working);
                  const t = active?.transform ?? identityTransform();
                  return (
                    <Group rotation={t.rotation} scaleX={t.scaleX} scaleY={t.scaleY} x={t.x} y={t.y}>
                      {maskOverlay ? (
                        <KonvaImage height={maskOverlay.height} image={maskOverlay} width={maskOverlay.width} x={0} y={0} />
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
                    </Group>
                  );
                })()}
              </Layer>
            ) : null}
            {(tool === "edit" || tool === "cutout") && maskMode && maskSubTool === "select" && selectDraft ? (
              // Live smart-select box preview (sc-3751), image-pixel coords like the crop rect.
              <Layer listening={false}>
                {(() => {
                  const t = activeLayerOf(working)?.transform ?? identityTransform();
                  return (
                    <Group rotation={t.rotation} scaleX={t.scaleX} scaleY={t.scaleY} x={t.x} y={t.y}>
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
                    </Group>
                  );
                })()}
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
            const { assets: imported, failureCount } = await importEditorReferenceFiles(files, importAsset);
            const ids = imported.map((asset) => asset.id);
            if (ids.length) {
              setRefAssetIds((prev) =>
                Array.from(new Set([...prev, ...ids])).slice(0, MAX_EDIT_REFERENCES - 1),
              );
            }
            if (failureCount > 0) {
              const added = ids.length
                ? `Added ${ids.length} reference image${ids.length === 1 ? "" : "s"}. `
                : "";
              setStatus({
                loading: false,
                error: `${added}Could not import ${failureCount} reference image${failureCount === 1 ? "" : "s"}.`,
              });
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
