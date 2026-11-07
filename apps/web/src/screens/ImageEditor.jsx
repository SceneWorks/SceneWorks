import { useMediaTicket } from "../hooks/useMediaTicket.js";
import {
  useMoveToolPanel,
  useTransformToolPanel,
  useCropToolPanel,
  useUpscaleToolPanel,
  useDetailToolPanel,
  useColorToolPanel,
  useCutoutToolPanel,
  useEditToolPanel,
  useBoxesToolPanel
} from "./imageEditor/useImageEditorPanels.js";
import { useEditorSave } from "./imageEditor/useEditorSave.js";
import { useEditorAiOps } from "./imageEditor/useEditorAiOps.js";
import { useLayerOps } from "./imageEditor/useLayerOps.js";
import { useEditorHistory } from "./imageEditor/useEditorHistory.js";
import { MIN_SCALE, MAX_SCALE, ZOOM_STEP, MIN_CROP_PX, EDITOR_LAYOUT_KEY, readStoredEditorLayout, EDITOR_TOOL_ORDER, EDITOR_TOOL_META, EDITOR_TOOL_ICONS, IeChevron, IeEyeOpen, IeEyeOff, EDITOR_SHORTCUTS } from "./imageEditor/metadata.jsx";
export { EDITOR_LAYOUTS, readStoredEditorLayout, EDITOR_TOOL_ORDER, EDITOR_TOOL_META, EDITOR_TOOL_ICONS } from "./imageEditor/metadata.jsx";
import { leaveGuardArming, closeConfirmMessage, saveStatusIndicator } from "./imageEditor/leaveGuard.js";
export { leaveGuardMessage, leaveGuardArming, closeConfirmMessage, saveStatusIndicator } from "./imageEditor/leaveGuard.js";
import { createEditorLaunchGuard } from "./imageEditor/launchGuard.js";
export { createEditorLaunchGuard } from "./imageEditor/launchGuard.js";
import { MAX_EDIT_REFERENCES, importEditorReferenceFiles, modelIsInpaintCapable } from "./imageEditor/references.js";
export { MAX_EDIT_REFERENCES, editReferenceIds, importEditorReferenceFiles, modelIsInpaintCapable } from "./imageEditor/references.js";

export { editedFilename, buildSaveProvenance } from "./imageEditor/provenance.js";
import { EDIT_OUTPUT_ASPECTS, clamp, cropRatioForKey, centeredCropRect, cropOverlayRects, BLANK_CANVAS_SIZES, blankCanvasDims } from "./imageEditor/geometry.js";
export { EDIT_OUTPUT_ASPECTS, editOutputAspectRatio, editOutputDims, cropRatioForKey, centeredCropRect, BLANK_CANVAS_MIN, BLANK_CANVAS_MAX, BLANK_CANVAS_SIZES, blankCanvasDims } from "./imageEditor/geometry.js";
import { compositeWorkingToCanvas, downloadSizeSentence, colorKeyMaskCanvasForImage, maskCutoutCanvasForImage, canvasToPngBlob, blobToEditorImage, revokeLayerUrls, collectCropLayers } from "./imageEditor/export.js";
export { EDITOR_CANVAS_MAX_SIDE, EDITOR_CANVAS_MAX_AREA, boundedEditorCanvasDimensions, compositeWorkingToCanvas, workingToPngFile, exportEditorFile, MAX_DESKTOP_EXPORT_BYTES, RASTER_REASON_EDITED, RASTER_REASON_DESKTOP_CAP, editorDownloadPlan, downloadSizeSentence, editorDownloadNote, collectCropLayers } from "./imageEditor/export.js";

export { HISTORY_LIMIT, emptyHistory, historyCheckpoint, historyUndo, historyRedo, canUndo, canRedo, runGuardedRestore } from "./imageEditor/history.js";
import { InlineLayerName } from "./imageEditor/InlineLayerName.jsx";
export { InlineLayerName } from "./imageEditor/InlineLayerName.jsx";
import { JOB_PATHS } from "../api/jobs.js";
import { mediaResponseWithTicketRetry } from "../api/media.js";
import { blobToImage } from "../imageDimensions.js";
import React, { useInsertionEffect, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Stage, Layer, Group, Image as KonvaImage, Line, Rect, Transformer } from "react-konva";
import { isAbortError } from "../api.js";

import { useAppContext } from "../context/AppContext.js";
import { useScreenActive } from "../context/ScreenActiveContext.js";
import { appConfirm } from "../appConfirm.jsx";

import { DEFAULT_MAC_CAPABILITIES, macFeatureBlock } from "../macGating.js";
import { assetUrl, assetCanRenderAsImage } from "../components/assetMedia.jsx";
// sc-24113 — the per-model ordered-reference ceiling. See `imageReferenceLimits.js`.
import { maxReferencesForModel } from "../imageReferenceLimits.js";
import { describeOpenedImage } from "../editorSourceWorkflow.js";

import { DatasetAddDialog } from "../components/DatasetAddDialog.jsx";

import { useLoraSelection } from "../components/LoraPickerField.jsx";
import { findModelEditLora, loraIsInstalled } from "../presetUtils.js";

import { mossAdvancedDefaults } from "../mossImageAdvanced.js";
import { mossAdvancedIssue } from "../mossImageAdvancedValidation.js";
import { isMossImage, mossEditDefaults, mossEditGenerationDefaults, mossComponentState, mossEditWireOptions, mossEditControlsError } from "../mossImage.js";
import { BLEND_MODES, activeLayerOf, drawLayerIntoCrop, identityTransform, layerById, replaceLayerBitmap, replaceLayerWithCroppedBitmap, singleLayerWorking } from "../imageLayers.js";

// Pure job builders + model/engine helpers (sc-6112) — extracted to a konva-free module
// so the Library batch flow can reuse them; imported for internal use and re-exported
// below to keep this module's public surface (and its tests) unchanged.
import { buildDetailJobBody, buildEditJobBody, buildUpscaleJobBody, detailCapableModels, editCapableModels, tileControlNetInstalled, tileControlNetModel, TILE_CONTROLNET_MODEL_ID, sam3SegmentInstalled, sam3SegmentModel, SAM3_SEGMENT_MODEL_ID, upscaleEngineHasSoftness, upscaleFactorsForEngine } from "../imageJobs.js";
import {
  availableUpscaleEngines as upscaleEnginesForPlatform,
  useUpscaleEngineFallback,
} from "../upscaleEngines.js";
import { IRIS_RESTORE_DEFAULTS } from "../irisRestore.js";

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
import { applyColorKeyToRgba, colorKeySelectionMaskRgba, documentPointToLayerPoint } from "./imageEditor/colorKeyMath.js";
import {
  ImageEditorToolPanel,
} from "./imageEditor/ImageEditorToolPanel.jsx";
import { EditorDepthPanel } from "./imageEditor/EditorDepthPanel.jsx";

import { editTransparencyFor, imageBytesCarryAlpha } from "../qwenAlpha.js";

export {
  buildDetailJobBody,
  buildEditJobBody,
  buildUpscaleJobBody,
  detailCapableModels,
  editCapableModels,
  tileControlNetInstalled,
  tileControlNetModel,
  TILE_CONTROLNET_MODEL_ID,
  sam3SegmentInstalled,
  sam3SegmentModel,
  SAM3_SEGMENT_MODEL_ID,
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
  colorKeySelectionMaskRgba,
  documentPointToLayerPoint,
};

export function ImageEditor() {
  useMediaTicket();
  const {
    activeProject,
    assets,
    characters,
    setPreviewAsset,
    token,
    requestedGpu,
    jobs,
    importAsset,
    confirmWorkflowEmbeddingBeforeGeneration,
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
  // Smart-select is cache-only at job time: backend capability and a complete
  // Model-Manager SAM3 install are both required before the UI offers a run.
  const smartSelectCapabilitySupported = macCapabilities?.features?.imageSegment?.supported === true;
  const smartSelectModel = sam3SegmentModel(models);
  const smartSelectInstalled = sam3SegmentInstalled(models);
  const smartSelectSupported = smartSelectCapabilitySupported && smartSelectInstalled;
  const [smartSelectDownloadRequested, setSmartSelectDownloadRequested] = useState(false);
  useEffect(() => {
    if (smartSelectInstalled) setSmartSelectDownloadRequested(false);
  }, [smartSelectInstalled]);
  const requestSmartSelectDownload = useCallback(() => {
    if (!smartSelectModel || !createModelDownloadJob) return;
    setSmartSelectDownloadRequested(true);
    createModelDownloadJob(smartSelectModel);
  }, [smartSelectModel, createModelDownloadJob]);

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
  useInsertionEffect(() => { editorLaunchRef.current = editorLaunch; }, [editorLaunch]);
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

  // Crop tool (sc-2430): client-side, rasterized into a new working image on Apply.
  const [tool, setTool] = useState("move");
  // Color-key cutout (sc-22462): selection/preview remain ephemeral until Apply,
  // then the active layer receives one alpha-only bitmap replacement.
  const [colorKeySeed, setColorKeySeed] = useState(null);
  const [colorKeyLayerId, setColorKeyLayerId] = useState(null);
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

  // Upscale tool (sc-2433): engine + factor for the in-flight request.
  const [upscaleEngine, setUpscaleEngine] = useState("real-esrgan");
  const [upscaleFactor, setUpscaleFactor] = useState(2);
  // SeedVR2 detail/softness knob (0..1, sc-4815) — only meaningful for the seedvr2 engine.
  const [upscaleSoftness, setUpscaleSoftness] = useState(0);
  // Engines offered in the picker; AuraSR is dropped on every platform (sc-3668 / sc-5499).
  // Iris restoration (sc-25683) joins only once its own task entry is installed (`models`).
  const availableUpscaleEngines = upscaleEnginesForPlatform(macCapabilities, models);
  // Iris restoration controls + the planner preview's status (a refused plan blocks the run).
  const [irisRestore, setIrisRestore] = useState(IRIS_RESTORE_DEFAULTS);
  const [irisPlanStatus, setIrisPlanStatus] = useState({ ready: false, blocked: false, error: null });
  // If the selected engine got gated out (e.g. a stale saved AuraSR selection), fall back to the
  // default real-esrgan engine (the guaranteed-available cross-platform upscaler) so the tool stays
  // usable. Shared with ImageStudio via the single fallback hook (sc-8853).
  useUpscaleEngineFallback({
    macCapabilities,
    models,
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
  const editModels = editCapableModels(imageModels, macCapabilities);
  const [editModel, setEditModel] = useState("");
  const [editPrompt, setEditPrompt] = useState("");
  const [editSeed, setEditSeed] = useState("");
  const [nativeEditResults, setNativeEditResults] = useState([]);
  // Guidance override (sc-10275): "" = use the edit model's default (shown as the
  // input placeholder); a finite value rides advanced.guidanceScale.
  const [editGuidance, setEditGuidance] = useState("");
  const [mossImageEdit, setMossImageEdit] = useState(mossEditDefaults);
  const [mossEditSettings, setMossEditSettings] = useState(() => ({ ...mossEditGenerationDefaults(), max_sequence_length: 4096, training_shift: 1, negativePrompt: "", mossImageAdvanced: mossAdvancedDefaults() }));
  const mossEdit = isMossImage(editModel);
  // Canvas-extend / outpaint (sc-2556): target output aspect (default "match" = the
  // working size) and how to fill it (crop trims, pad bars, outpaint generates).
  const [editAspect, setEditAspect] = useState("match");
  const [editFitMode, setEditFitMode] = useState("crop");
  // Transparency on the edit lane (sc-24114). `null` = follow the working image: ON when it
  // carries alpha, so an AI edit of an RGBA source keeps its alpha. An explicit toggle wins until
  // the next image is opened.
  const [workingHasAlpha, setWorkingHasAlpha] = useState(false);
  const [editTransparentChoice, setEditTransparentChoice] = useState(null);
  const editTransparent = editTransparencyFor(editTransparentChoice, workingHasAlpha);

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
    const caps = editCapableModels(imageModels, macCapabilities);
    if (caps.length && !caps.some((model) => model.id === editModel)) setEditModel(caps[0].id);
  }, [imageModels, editModel, macCapabilities]);

  // Same default/self-heal for the detail backbone.
  useEffect(() => {
    const caps = detailCapableModels(imageModels);
    if (caps.length && !caps.some((model) => model.id === detailModel)) setDetailModel(caps[0].id);
  }, [imageModels, detailModel]);

  // The chosen edit model + whether it accepts an inpaint mask (gates the mask tool).
  const selectedEditModel = editModels.find((model) => model.id === editModel) ?? null;
  const mossEditReady = !mossEdit || mossComponentState(selectedEditModel, null, "edit_image").ready;
  let mossEditError = "";
  if (mossEdit) {
    mossEditError = mossEditControlsError(mossEditSettings);
    mossEditError ||= mossAdvancedIssue(mossEditSettings.mossImageAdvanced, { assets, prompt: editPrompt, prompts: mossEditSettings.promptBatch ?? [], negativePrompt: mossEditSettings.negativePrompt, width: mossEditSettings.width, height: mossEditSettings.height, context: mossEditSettings.max_sequence_length });
    try { mossEditWireOptions(mossImageEdit); } catch (error) { mossEditError = error.message; }
    if (![mossEditSettings.width, mossEditSettings.height].every((size) => Number.isInteger(size) && size >= 16 && size <= 4096 && size % 16 === 0)) mossEditError = "MOSS edit dimensions must be multiples of 16 in 16–4096.";
    if (refAssetIds.length > 3 || refAssetIds.some((id) => !(assets ?? []).some((asset) => asset.id === id && assetCanRenderAsImage(asset)))) mossEditError = "An ordered MOSS reference is missing or exceeds four total slots. Reacquire it before editing.";
  }
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
  // How many references THIS model takes (sc-24113). The cap used to be one module constant applied
  // to every multi-reference model in the catalog — `MAX_EDIT_REFERENCES = 4`, mirroring the
  // worker's FLUX.2 activation budget. Qwen-Image 2.1 takes ten, so the ceiling became a per-model
  // reading of `limits.maxReferenceAssets`, the key the video lane has declared since sc-17160.
  //
  // `MAX_EDIT_REFERENCES` remains the fallback rather than being replaced, so every model that
  // declares nothing keeps exactly the cap it had and nothing about FLUX.2 / SenseNova / Krea moves.
  const maxEditReferences = maxReferencesForModel(selectedEditModel, MAX_EDIT_REFERENCES);
  // Drop any attached references when the model can't use them (switched away from a multiReference
  // model), so a stale selection never rides a job that would ignore it.
  useEffect(() => {
    if (!multiRefCapable && refAssetIds.length) setRefAssetIds([]);
  }, [multiRefCapable, refAssetIds.length]);
  // ... and TRIM rather than drop when the new model simply takes fewer. Switching from Qwen 2.1
  // (10) to FLUX.2 (4) with six attached must not silently send six to a model that would truncate
  // them: the rail shows what will actually be sent. Trimming from the END keeps the ordering the
  // user built, which is the half that carries meaning.
  useEffect(() => {
    if (multiRefCapable && refAssetIds.length > maxEditReferences - 1) {
      setRefAssetIds((prev) => prev.slice(0, maxEditReferences - 1));
    }
  }, [multiRefCapable, maxEditReferences, refAssetIds.length]);

  // Save / export (sc-2434). `dirty` tracks edits not yet persisted to the Library;
  // `edits` is the ordered provenance chain; `savedAssetId` flags a completed Save
  // for the bar's "Saved" hint. A fresh open clears all three.
  const [dirty, setDirty] = useState(false);
  const [edits, setEdits] = useState([]);

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

  // Live mirror of the working document for synchronous reads inside the commit
  // handlers (a checkpoint captures the pre-operation stack; restore reuses the
  // live decoded images for unchanged layers and revokes the URLs it drops).
  const workingRef = useRef(null);
  const mountedRef = useRef(false);
  const openGeneration = useRef(0);
  const openController = useRef(null);
  const invalidateOpen = useCallback(() => {
    openGeneration.current += 1;
    openController.current?.abort();
    openController.current = null;
    return openGeneration.current;
  }, []);
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
    smartSelectSupported,
    aiOp,
    activeProject,
    requestedGpu,
    runAiOp: runAiOpBridge,
    stagePointToImage: stagePointToImageBridge,
    stagePointToActiveLayerImage,
    getWorking: () => workingRef.current,
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
    maskSource,
    maskTargetLayerId,
    maskCoordinateSpace,
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
    installMaskBaseCanvas,
    refineMask,
    resetMaskState,
  } = maskTool;

  const colorKeyActiveLayer = activeLayerOf(working);
  const colorKeyActiveLayerId = colorKeyActiveLayer?.id ?? null;
  const colorKeyActiveLayerImage = colorKeyActiveLayer?.image ?? null;

  // Color key is an input to the shared editable mask, not a separate apply
  // path. Changing Connected/Global/tolerance/softness regenerates the base;
  // brush/erase and geometric refinements then operate on that same selection.
  useEffect(() => {
    if (
      !colorKeySeed ||
      !colorKeyActiveLayerImage ||
      colorKeyActiveLayerId !== colorKeyLayerId ||
      tool !== "cutout"
    ) return;
    try {
      const mask = colorKeyMaskCanvasForImage(colorKeyActiveLayerImage, colorKeySeed, {
        tolerance: colorKeyTolerance,
        softness: colorKeySoftness,
        global: colorKeyGlobal,
      });
      installMaskBaseCanvas(mask, { source: "colorKey", targetLayerId: colorKeyActiveLayerId });
      setMaskMode(true);
      setMaskSubTool("brush");
      setCutoutKeepSelected(false);
    } catch {
      clearMask();
    }
  }, [
    tool,
    colorKeyActiveLayerId,
    colorKeyActiveLayerImage,
    colorKeySeed,
    colorKeyLayerId,
    colorKeyTolerance,
    colorKeySoftness,
    colorKeyGlobal,
    installMaskBaseCanvas,
    clearMask,
    setMaskMode,
    setMaskSubTool,
  ]);

  // Do not let an eyedropper seed from one layer regenerate after another
  // layer becomes active; the hook independently clears its bound mask state.
  useEffect(() => {
    if (colorKeyLayerId && working?.activeLayerId !== colorKeyLayerId) {
      setColorKeySeed(null);
      setColorKeyLayerId(null);
    }
  }, [working?.activeLayerId, colorKeyLayerId]);

  // SAM3 cutout preview is an offscreen alpha composition, just like color key:
  // refinement remains editable and no document/history mutation occurs until Apply.
  const maskCutoutPreview = useMemo(() => {
    const layer = activeLayerOf(working);
    if (
      tool !== "cutout" ||
      !maskMode ||
      !layer?.image ||
      maskTargetLayerId !== layer.id ||
      (!maskHasContent(maskLines) && !maskBaseImage)
    ) return null;
    try {
      return {
        layerId: layer.id,
        image: maskCutoutCanvasForImage(layer.image, rasterizeMaskToCanvas(), cutoutKeepSelected),
      };
    } catch {
      return null;
    }
  }, [working, tool, maskMode, maskTargetLayerId, maskLines, maskBaseImage, rasterizeMaskToCanvas, cutoutKeepSelected]);

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
      invalidateOpen();
      revokeLayerUrls(workingRef.current?.layers);
    };
  }, [invalidateOpen]);

  useEffect(() => {
    setStatus((current) => current.loading ? { ...current, loading: false } : current);
    return () => { invalidateOpen(); };
  }, [activeProject?.id, invalidateOpen]);

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
    setColorKeyLayerId(null);
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
  // reproduces the pre-layers single-bitmap behavior; the layer controls rendered
  // by this editor create and manage multi-layer stacks (sc-6118).
  const installWorkingImage = useCallback(
    (image, objectUrl, blob, source) => {
      revokeLayerUrls(workingRef.current?.layers);
      needsFitRef.current = true;
      resetEditorOverlays();
      setWorking(singleLayerWorking({ id: nextLayerId(), image, objectUrl, blob, source }));
    },
    [resetEditorOverlays],
  );

  const {
    historyFlags,
    checkpoint,
    resetHistory,
    undo,
    redo
  } = useEditorHistory({
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
  });

  // ── Keyboard shortcuts (sc-6111) ───────────────────────────────────────────
  // One editor-scoped window keydown handler. Held behind a ref so the listener is
  // subscribed once (no add/remove churn during the high-frequency re-renders of a
  // crop / box / mask drag) while always seeing the latest tool + selection state.
  // Never fires while a text field is focused, so typing a prompt / box description
  // / renaming a layer is left to the browser. Undo/redo (sc-6106) are the only
  // modified combos we own; the rest are single keys that mirror the toolbar and the
  // zoom bar. `?` toggles the quick reference and works before an image is open.
  const onEditorKeyDownRef = useRef(null);
  const onEditorKeyDown = (event) => {
    // Under selective keep-alive (sc-11959) this editor stays mounted and this window
    // listener stays subscribed even when another view is foregrounded, so gate every
    // shortcut on the active flag — a backgrounded editor must never undo/redo, delete a
    // box, switch tools, or toggle the reference in response to keys meant for the visible
    // screen (sc-13589). The handler is refreshed after each commit, so `screenActive` is current.
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

  useInsertionEffect(() => { onEditorKeyDownRef.current = onEditorKeyDown; }, [onEditorKeyDown]);

  useEffect(() => {
    const handler = (event) => onEditorKeyDownRef.current?.(event);
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const openFromBlob = useCallback(
    async (blob, source, generation = invalidateOpen()) => {
      const isCurrent = () => mountedRef.current && generation === openGeneration.current;
      const controller = openController.current ?? new AbortController();
      openController.current = controller;
      let prepared;
      setStatus({ loading: true, error: "" });
      try {
        prepared = await blobToEditorImage(blob, { signal: controller.signal });
        // Keep the bytes we opened (sc-15954). The blob is retained by reference — no copy — and
        // `installedBlob` is the identity token `editorDownloadPlan` compares the layer stack
        // against. Note it is `prepared.blob`, not `blob`: a source over EDITOR_CANVAS_MAX_SIDE is
        // resampled on load, so the two differ exactly in the case the story calls the sharp edge,
        // and the passthrough branch must ship the ORIGINAL rather than the proxy the canvas is
        // showing. Whether those bytes carry a recipe is NOT decided here — `workflow` is a
        // starting state and the effect below asks the one reader.
        const opened = await describeOpenedImage(blob, source?.name);
        // sc-24114: whether the ORIGINAL bytes carry alpha — defaults the edit's transparency.
        let hasAlpha = false;
        try {
          hasAlpha = imageBytesCarryAlpha(new Uint8Array(await blob.arrayBuffer()));
        } catch {
          hasAlpha = false;
        }
        if (!isCurrent()) {
          URL.revokeObjectURL(prepared.objectUrl);
          return false;
        }
        setWorkingHasAlpha(hasAlpha);
        setEditTransparentChoice(null);
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
        return true;
      } catch (err) {
        if (prepared?.objectUrl) URL.revokeObjectURL(prepared.objectUrl);
        if (isCurrent()) setStatus({ loading: false, error: err.message || "Could not open image" });
        return false;
      }
    },
    [installWorkingImage, resetHistory, invalidateOpen],
  );

  const openAsset = useCallback(
    async (assetId) => {
      const generation = invalidateOpen();
      const controller = new AbortController();
      openController.current = controller;
      const isCurrent = () => mountedRef.current && generation === openGeneration.current;
      const asset = imageAssets.find((item) => item.id === assetId);
      if (!asset) return;
      const url = assetUrl(asset);
      if (!url) {
        setStatus({ loading: false, error: "Asset has no media file" });
        return;
      }
      setStatus({ loading: true, error: "" });
      try {
        const res = await mediaResponseWithTicketRetry(() => assetUrl(asset), { signal: controller.signal });
        if (!res.ok) throw new Error(`Failed to load asset (${res.status})`);
        const blob = await res.blob();
        if (!isCurrent()) return;
        await openFromBlob(blob, {
          kind: "asset",
          assetId: asset.id,
          name: asset.displayName ?? asset.id,
        }, generation);
      } catch (err) {
        if (isCurrent() && !isAbortError(err)) setStatus({ loading: false, error: err.message || "Could not load asset" });
      }
    },
    [imageAssets, openFromBlob, invalidateOpen],
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
      const generation = invalidateOpen();
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      let blob;
      try {
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, width, height);
        blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      } finally {
        canvas.width = 0;
        canvas.height = 0;
      }
      if (!mountedRef.current || generation !== openGeneration.current) return;
      if (!blob) {
        setStatus({ loading: false, error: "Could not create the canvas." });
        return;
      }
      if (await openFromBlob(blob, { kind: "blank", name: "Untitled layout" }, generation)) setTool("boxes");
    },
    [openFromBlob, invalidateOpen],
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
      setColorKeyLayerId(null);
      clearMask();
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
          (blob) => {
            canvas.width = 0;
            canvas.height = 0;
            if (blob) resolve(new File([blob], filename || `${base}.png`, { type: "image/png" }));
            else reject(new Error("Could not encode the layer."));
          },
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
  useInsertionEffect(() => { replaceLayerImageRef.current = replaceLayerImage; }, [replaceLayerImage]);

  // Commit the currently refined color-key or SAM3 mask to its originating
  // active layer only. The mask is
  // converted to an alpha multiplier, so keep/remove choices both preserve an
  // already-transparent source pixel and produce one undoable bitmap replacement.
  const applyMaskCutout = useCallback(async () => {
    const work = workingRef.current;
    const layer = activeLayerOf(work);
    if (
      !layer?.image ||
      maskTargetLayerId !== layer.id ||
      (!maskHasContent(maskLines) && !maskBaseImage)
    ) return;
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
      const edit = maskSource === "colorKey"
        ? {
            op: "colorKey",
            mode: colorKeyGlobal ? "global" : "connected",
            tolerance: colorKeyTolerance,
            softness: colorKeySoftness,
            refined: true,
          }
        : { op: "sam3Cutout", mode: cutoutKeepSelected ? "keepSelected" : "removeSelected" };
      setEdits((prev) => [...prev, edit]);
      setDirty(true);
      clearMask();
      setColorKeySeed(null);
      setColorKeyLayerId(null);
      setTool("move");
    } catch (err) {
      setStatus({ loading: false, error: err.message || "Could not apply the selection cutout." });
    }
  }, [
    maskLines,
    maskBaseImage,
    maskTargetLayerId,
    maskSource,
    rasterizeMaskToCanvas,
    cutoutKeepSelected,
    colorKeyGlobal,
    colorKeyTolerance,
    colorKeySoftness,
    checkpoint,
    replaceLayerImage,
    clearMask,
  ]);

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
    const generation = openGeneration.current;
    let cropped;
    try {
      cropped = await collectCropLayers(working.layers, async (layer) => {
        const canvas = document.createElement("canvas");
        canvas.width = sw;
        canvas.height = sh;
        try {
          const ctx = canvas.getContext("2d");
          drawLayerIntoCrop(ctx, layer, { x: sx, y: sy, width: sw, height: sh }, straighten);
          const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
          if (!blob) throw new Error("Could not encode the crop.");
          const { image, objectUrl } = await blobToImage(blob);
          return { id: layer.id, image, objectUrl, blob };
        } finally {
          canvas.width = 0;
          canvas.height = 0;
        }
      }, () => mountedRef.current && workingRef.current === working && generation === openGeneration.current);
    } catch (err) {
      setStatus({ loading: false, error: err?.message || "Could not crop the layers." });
      return;
    }
    if (!cropped) return;
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

  const {
    layerOpacityGestureRef,
    selectLayer,
    toggleLayerVisible,
    changeLayerOpacity,
    renameLayer,
    reorderLayer,
    deleteLayer,
    duplicateLayerById,
    addBlankLayer,
    setLayerBlend,
    startTransform,
    commitActiveLayerTransform,
    resetActiveLayerTransform,
    setActiveTransform,
    onTransformSlider,
    endTransformGesture,
    flipActiveLayer
  } = useLayerOps({
    setWorking,
  workingRef,
  checkpoint,
  setDirty,
  nextLayerId,
  setStatus,
  working,
  setTool,
  imageNodeRef
  });

  // The module-level `compositeWorkingToCanvas`, bound to the current document. Kept as a thin
  // wrapper so the many call sites below read unchanged while the flatten itself is exported and
  // directly testable (sc-24111).
  function compositeToCanvas(work = working) {
    return compositeWorkingToCanvas(work);
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
    if (!mossEditReady || mossEditError) return;
    const prompt = mossEdit ? editPrompt : editPrompt.trim();
    let sourceFile;
    try {
      sourceFile = await bakeBoxesToFile();
    } catch (err) {
      setStatus({ loading: false, error: `Could not bake boxes: ${err.message || err}` });
      return;
    }
    runAiOp({
      label: "edit",
      endpoint: JOB_PATHS.image,
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
          mossImageEdit,
          mossEditSettings,
          referenceAssetIds: mossEdit ? [scratch.id, ...refAssetIds] : null,
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
  useInsertionEffect(() => { stagePointToImageRef.current = stagePointToImage; }, [stagePointToImage]);

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
    setColorKeyLayerId(layer.id);
  }

  const {
    workingImageToFile,
    runAiOp,
    runUpscale,
    runDetail,
    runEdit
  } = useEditorAiOps({
    working,
  aiOp,
  activeProject,
  confirmFlatten,
  setStatus,
  confirmWorkflowEmbeddingBeforeGeneration,
  mountedRef,
  activeLayerToFile,
  importAsset,
  purgeAsset,
  token,
  trackEditorScratchOp,
  setAiOp,
  setTool,
  runAiOpRef,
  upscaleEngine,
  irisPlanStatus,
  irisRestore,
  requestedGpu,
  upscaleFactor,
  upscaleSoftness,
  detailModel,
  detailStrength,
  detailCnScale,
  mossEdit,
  editPrompt,
  mossEditSettings,
  editModel,
  editLoraRequiredMissing,
  mossEditReady,
  mossEditError,
  editFitMode,
  canMask,
  editAspect,
  maskLines,
  maskBaseImage,
  rasterizeMaskToFile,
  editTransparent,
  selectedEditModel,
  multiRefCapable,
  refAssetIds,
  maxEditReferences,
  editSeed,
  editLoraSelection,
  managedEditLoraId,
  editLora,
  editGuidance,
  mossImageEdit,
  jobs,
  setNativeEditResults,
  checkpoint,
  workingRef,
  replaceLayerImage,
  installWorkingImage,
  setEdits,
  setDirty,
  releaseEditorScratchOp
  });

  const {
    saving,
    runSave,
    downloadPlan,
    downloadNote,
    runDownload
  } = useEditorSave({
    working,
  setStatus,
  workingImageToFile,
  importAsset,
  edits,
  setSavedAssetId,
  setDirty,
  activeProject,
  token
  });

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
    invalidateOpen();
    revokeLayerUrls(workingRef.current?.layers);
    workingRef.current = null;
    setWorking(null);
    setEdits([]);
    setDirty(false);
    setSavedAssetId(null);
    setAiOp(null);
    resetHistory();
    setStatus({ loading: false, error: "" });
  }, [resetHistory, invalidateOpen]);

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
      ? (maskSubTool === "select" ? "Drag a box to select an object with SAM3" : "Paint to refine the cutout selection")
      : (colorKeySeed ? "Adjust the key, then apply the alpha cutout" : "Click the background color to preview a cutout"),
    edit: maskMode ? "Paint or box-select the region to edit" : "Describe the edit on the right",
    boxes: "Drag to draw a region, then describe it",
    depth: "Pick a processing resolution, then estimate depth",
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

  const movePanel = useMoveToolPanel({
    working,
    layerCount,
    fitToView,
    actualSize
  });
  const transformPanel = useTransformToolPanel({
    working,
    setActiveTransform,
    endTransformGesture,
    onTransformSlider,
    flipActiveLayer,
    resetActiveLayerTransform,
    setTool
  });
  const cropPanel = useCropToolPanel({
    ratioKey,
    rotated,
    cropRect,
    straighten,
    chooseRatio,
    toggleRotate,
    setCropDim,
    setStraighten,
    applyCrop,
    cancelCrop
  });
  const upscalePanel = useUpscaleToolPanel({
    availableUpscaleEngines,
    upscaleEngine,
    upscaleFactor,
    aiOp,
    working,
    token,
    irisRestore,
    upscaleSoftness,
    irisPlanStatus,
    setUpscaleEngine,
    setUpscaleFactor,
    setIrisRestore,
    setIrisPlanStatus,
    setUpscaleSoftness,
    runUpscale,
    setTool
  });
  const detailPanel = useDetailToolPanel({
    detailModels,
    detailModel,
    selectedDetailModel,
    detailStrength,
    detailCnScale,
    tileControlNetReady,
    tileControlNet,
    tileControlNetDownloadRequested,
    aiOp,
    createModelDownloadJob,
    setDetailModel,
    setDetailStrength,
    setDetailCnScale,
    requestTileControlNetDownload,
    runDetail,
    setTool
  });
  const colorPanel = useColorToolPanel({
    colorMode,
    colorAdjust,
    colorChannel,
    histogramRef,
    levels,
    curves,
    channelStroke,
    activeGradeIsIdentity,
    setColorMode,
    setAdjustValue,
    resetAdjust,
    setColorChannel,
    setLevelsValue,
    setCurves,
    resetActiveColorMode,
    applyColorGrade
  });
  const cutoutPanel = useCutoutToolPanel({
    colorKeySeed,
    colorKeyGlobal,
    colorKeyTolerance,
    colorKeySoftness,
    smartSelectSupported,
    maskMode,
    maskSubTool,
    aiOp,
    smartSelectCapabilitySupported,
    smartSelectModel,
    smartSelectDownloadRequested,
    maskBrush,
    maskErase,
    maskRefineRadius,
    maskLines,
    maskBaseImage,
    cutoutKeepSelected,
    maskSource,
    setColorKeyGlobal,
    setColorKeyTolerance,
    setColorKeySoftness,
    setColorKeySeed,
    clearMask,
    setCutoutKeepSelected,
    setMaskMode,
    setMaskSubTool,
    setMaskErase,
    requestSmartSelectDownload,
    setMaskBrush,
    setMaskRefineRadius,
    refineMask,
    applyMaskCutout,
    setTool
  });
  const editPanel = useEditToolPanel({
    mossEdit,
    mossEditReady,
    mossEditError,
    mossImageEdit,
    mossEditSettings,
    token,
    jobs,
    maxEditReferences,
    aiOp,
    canMask,
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
    imageAssets,
    maskActive,
    maskBaseImage,
    maskBrush,
    maskErase,
    maskLines,
    maskMode,
    maskRefineRadius,
    maskSubTool,
    multiRefCapable,
    refAssetIds,
    editTransparent,
    selectedEditLoras,
    selectedEditModel,
    showIncompatibleEditLoras,
    smartSelectCapabilitySupported,
    smartSelectDownloadRequested,
    smartSelectModel,
    smartSelectSupported,
    tensorAssets: assets,
    characters,
    projectId: activeProject?.id,
    createLoraDownloadJob,
    createModelDownloadJob,
    setMossImageEdit,
    setMossEditSettings,
    clearMask,
    refineMask,
    setEditTransparent: setEditTransparentChoice,
    requestEditLoraDownload,
    requestSmartSelectDownload,
    runEdit,
    setEditAspect,
    setEditFitMode,
    setEditGuidance,
    setEditModel,
    setEditPrompt,
    setEditSeed,
    setMaskBrush,
    setMaskErase,
    setMaskMode,
    setMaskRefineRadius,
    setMaskSubTool,
    setRefAssetIds,
    setRefPickerOpen,
    setShowIncompatibleEditLoras,
    importTensorAsset: importAsset
  });
  const boxesPanel = useBoxesToolPanel({
    mossEdit,
    mossEditReady,
    mossEditError,
    mossImageEdit,
    mossEditSettings,
    token,
    jobs,
    imageAssets,
    refAssetIds,
    maxEditReferences,
    aiOp,
    boxColor,
    boxes,
    composedPrompt,
    editModel,
    editModels,
    editPrompt,
    selectedBox,
    selectedBoxGaps,
    selectedBoxId,
    selectedEditModel,
    tensorAssets: assets,
    characters,
    projectId: activeProject?.id,
    createModelDownloadJob,
    updateBox,
    setMossImageEdit,
    setMossEditSettings,
    setRefAssetIds,
    chooseBoxColor,
    clearBoxes,
    deleteBox,
    runBoxEdit,
    setEditModel,
    setEditPrompt,
    setSelectedBoxId,
    importTensorAsset: importAsset
  });
  const toolPanels = { move: movePanel, transform: transformPanel, crop: cropPanel, upscale: upscalePanel, detail: detailPanel, color: colorPanel, cutout: cutoutPanel, edit: editPanel, boxes: boxesPanel };
  const renderToolPanel = (key) =>
    key === "depth" ? (
      <EditorDepthPanel dirty={dirty} working={working} />
    ) : (
      <ImageEditorToolPanel panelKey={key} {...toolPanels[key]} />
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
        {nativeEditResults.length ? <section className="ie-native-results" aria-label="Native edit outputs">
          <p>This job produced {nativeEditResults.length} {nativeEditResults[0].type === "tensor" ? "tensor outputs" : "images"}. The outputs and edit inputs are available in Library for download and replay. {nativeEditResults[0].type === "tensor" ? "Tensor output leaves the canvas unchanged." : "The first image is applied to the canvas."}</p>
          {nativeEditResults.map((asset, index) => <span key={asset.id}><a href={assetUrl(asset)} download>Output {index + 1}: {asset.type === "tensor" ? "FP32 NPY tensor" : "image"}</a>{asset.extra?.workflowCompanion?.path ? <a href={assetUrl({ projectId: asset.projectId, file: { path: asset.extra.workflowCompanion.path } })} download>Portable workflow</a> : null}</span>)}
        </section> : null}
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
                      : layer.image}
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
              // Cutout masks are active-bitmap coordinates and mirror its transform;
              // AI Edit masks remain document coordinates to align with composite
              // staging and outpaint, including transformed/multi-layer documents.
              <Layer listening={false}>
                {(() => {
                  const active = activeLayerOf(working);
                  const t = maskCoordinateSpace === "layer"
                    ? active?.transform ?? identityTransform()
                    : identityTransform();
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
                  const t = maskCoordinateSpace === "layer"
                    ? activeLayerOf(working)?.transform ?? identityTransform()
                    : identityTransform();
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
              Array.from(new Set([...prev, ...ids])).slice(0, maxEditReferences - 1),
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
                Array.from(new Set([...prev, ...ids])).slice(0, maxEditReferences - 1),
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
