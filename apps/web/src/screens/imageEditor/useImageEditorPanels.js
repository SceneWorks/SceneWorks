import { useEditorPanelActions } from "./useEditorPanelActions.js";

// Each tool owns an explicit producer interface. Only event actions are stable
// committed bridges; render data and derived helpers remain current props.

export function useMoveToolPanel({
  working,
  layerCount,
  fitToView,
  actualSize
}) {
  const actions = useEditorPanelActions({ fitToView, actualSize });
  return { working, layerCount, ...actions };
}

export function useTransformToolPanel({
  working,
  setActiveTransform,
  endTransformGesture,
  onTransformSlider,
  flipActiveLayer,
  resetActiveLayerTransform,
  setTool
}) {
  const actions = useEditorPanelActions({ setActiveTransform, endTransformGesture, onTransformSlider, flipActiveLayer, resetActiveLayerTransform, setTool });
  return { working, ...actions };
}

export function useCropToolPanel({
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
}) {
  const actions = useEditorPanelActions({ chooseRatio, toggleRotate, setCropDim, setStraighten, applyCrop, cancelCrop });
  return { ratioKey, rotated, cropRect, straighten, ...actions };
}

export function useUpscaleToolPanel({
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
}) {
  const actions = useEditorPanelActions({ setUpscaleEngine, setUpscaleFactor, setIrisRestore, setIrisPlanStatus, setUpscaleSoftness, runUpscale, setTool });
  return { availableUpscaleEngines, upscaleEngine, upscaleFactor, aiOp, working, token, irisRestore, upscaleSoftness, irisPlanStatus, ...actions };
}

export function useDetailToolPanel({
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
}) {
  const actions = useEditorPanelActions({ createModelDownloadJob, setDetailModel, setDetailStrength, setDetailCnScale, requestTileControlNetDownload, runDetail, setTool });
  return { detailModels, detailModel, selectedDetailModel, detailStrength, detailCnScale, tileControlNetReady, tileControlNet, tileControlNetDownloadRequested, aiOp, ...actions };
}

export function useColorToolPanel({
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
}) {
  const actions = useEditorPanelActions({ setColorMode, setAdjustValue, resetAdjust, setColorChannel, setLevelsValue, setCurves, resetActiveColorMode, applyColorGrade });
  return { colorMode, colorAdjust, colorChannel, histogramRef, levels, curves, channelStroke, activeGradeIsIdentity, ...actions };
}

export function useCutoutToolPanel({
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
}) {
  const actions = useEditorPanelActions({ setColorKeyGlobal, setColorKeyTolerance, setColorKeySoftness, setColorKeySeed, clearMask, setCutoutKeepSelected, setMaskMode, setMaskSubTool, setMaskErase, requestSmartSelectDownload, setMaskBrush, setMaskRefineRadius, refineMask, applyMaskCutout, setTool });
  return { colorKeySeed, colorKeyGlobal, colorKeyTolerance, colorKeySoftness, smartSelectSupported, maskMode, maskSubTool, aiOp, smartSelectCapabilitySupported, smartSelectModel, smartSelectDownloadRequested, maskBrush, maskErase, maskRefineRadius, maskLines, maskBaseImage, cutoutKeepSelected, maskSource, ...actions };
}

export function useEditToolPanel({
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
  tensorAssets,
  characters,
  projectId,
  createLoraDownloadJob,
  createModelDownloadJob,
  setMossImageEdit,
  setMossEditSettings,
  clearMask,
  refineMask,
  setEditTransparent,
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
  importTensorAsset
}) {
  const actions = useEditorPanelActions({ createLoraDownloadJob, createModelDownloadJob, setMossImageEdit, setMossEditSettings, clearMask, refineMask, setEditTransparent, requestEditLoraDownload, requestSmartSelectDownload, runEdit, setEditAspect, setEditFitMode, setEditGuidance, setEditModel, setEditPrompt, setEditSeed, setMaskBrush, setMaskErase, setMaskMode, setMaskRefineRadius, setMaskSubTool, setRefAssetIds, setRefPickerOpen, setShowIncompatibleEditLoras, importTensorAsset });
  return { mossEdit, mossEditReady, mossEditError, mossImageEdit, mossEditSettings, token, jobs, maxEditReferences, aiOp, canMask, editAspect, editFitMode, editGuidance, editLora, editLoraDownloadRequested, editLoraInstalled, editLoraRequiredMissing, editLoraSelection, editModel, editModels, editPrompt, editSeed, editorPickerLoras, imageAssets, maskActive, maskBaseImage, maskBrush, maskErase, maskLines, maskMode, maskRefineRadius, maskSubTool, multiRefCapable, refAssetIds, editTransparent, selectedEditLoras, selectedEditModel, showIncompatibleEditLoras, smartSelectCapabilitySupported, smartSelectDownloadRequested, smartSelectModel, smartSelectSupported, tensorAssets, characters, projectId, ...actions };
}

export function useBoxesToolPanel({
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
  tensorAssets,
  characters,
  projectId,
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
  importTensorAsset
}) {
  const actions = useEditorPanelActions({ createModelDownloadJob, updateBox, setMossImageEdit, setMossEditSettings, setRefAssetIds, chooseBoxColor, clearBoxes, deleteBox, runBoxEdit, setEditModel, setEditPrompt, setSelectedBoxId, importTensorAsset });
  return { mossEdit, mossEditReady, mossEditError, mossImageEdit, mossEditSettings, token, jobs, imageAssets, refAssetIds, maxEditReferences, aiOp, boxColor, boxes, composedPrompt, editModel, editModels, editPrompt, selectedBox, selectedBoxGaps, selectedBoxId, selectedEditModel, tensorAssets, characters, projectId, ...actions };
}
