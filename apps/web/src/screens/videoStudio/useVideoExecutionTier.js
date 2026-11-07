import { SCAIL2_CANDLE_PRODUCT_TIERS, TIER_SCREEN, videoExecutionTierModel, unavailableRecipeTierMessage } from "./videoTierHelpers.js";
import { candleVideoTierPickerAvailable } from "../../modelEligibility.js";
import { useCallback, useEffect, useMemo } from "react";
import { useQuantTierPicker } from "../../components/generationStudio.jsx";
import { SCAIL2_MODEL_ID } from "../../videoStudioValidation.js";
import { videoModelServesMode } from "../../modelEligibility.js";
import { allPossibleTiers, installedTiers, tierLabel, tierPickerOptions, tierQuantize } from "../../quantTier.js";
import { suggestTier } from "../../registryTierEvidence.js";
import { useHostMemory } from "../../hooks/useHostMemory.js";
import { hostMemoryGbForBackend } from "../../hostMemory.js";

// Extracted from VideoStudio along its existing concern boundary.
export function useVideoExecutionTier({
  macGating,
  selectedModel,
  model,
  preferencesHydrated,
  recipeTierRequest,
  videoModels,
  macVideoModels,
  mode,
  macCapabilities,
  setModel,
  setRecipeTierRequest,
  setRecipeModelNotice,
  supportsQuantization
}) {
  // Sampler / scheduler menus declared by the model. Video Wan torch
  // declares the full menu; sealed paths (LTX native, MLX) drop to
  // default-only and the picker hides. Gated to the ACTIVE backend (epic 7114 P5):
  // `macGating` is the worker `mlx_required` master switch, so the menu reflects the
  // manifest's `mlx.limits` override on Mac/MLX and `candle.limits` on the candle build.
  const activeBackend = macGating ? "mlx" : "candle";

  // The same hosted q4/q8/bf16 matrices drive both native video backends. Candle Wan tiers must use
  // this explicit picker too: sending the unrelated Torch GGUF `advanced.quantization` field to the
  // Candle worker would otherwise be ignored and silently replaced by its default tier.
  const nativeTierLane =
    activeBackend === "mlx" ||
    (activeBackend === "candle" && candleVideoTierPickerAvailable(selectedModel));

  const scail2CandleTierLane =
    activeBackend === "candle" && selectedModel?.id === SCAIL2_MODEL_ID;

  // A Video Studio-only projection: retain the complete catalog object for every non-tier concern,
  // but narrow every tier vocabulary the shared picker understands. The original model object still
  // reaches Model Manager unchanged, so q4/q8 install and repair metadata remain visible there.
  const executionTierModel = useMemo(
    () => videoExecutionTierModel(selectedModel, activeBackend),
    [selectedModel, activeBackend],
  );

  const tierOptions = useMemo(
    () => ({ convRotEligible: false, nvfp4Eligible: false }),
    [],
  );

  const availableTiers = useMemo(
    () => (nativeTierLane ? installedTiers(executionTierModel, tierOptions) : []),
    [nativeTierLane, executionTierModel, tierOptions],
  );

  // The full display set (all possible tiers, installed or not) + the picker option list with
  // un-downloaded tiers disabled — same show-all/disable-unavailable rule as Image Studio. `availableTiers`
  // stays the SELECTABLE/send set. Ordinarily the picker needs more than one possible tier; the
  // explicit SCAIL-2 Candle tier surface remains visible so users can see which shipped packages
  // are admitted even when a particular local install is incomplete.
  const possibleTiers = useMemo(
    () => (nativeTierLane ? allPossibleTiers(executionTierModel, tierOptions) : []),
    [nativeTierLane, executionTierModel, tierOptions],
  );

  const tierPickerItems = useMemo(
    () => (nativeTierLane ? tierPickerOptions(executionTierModel, tierOptions) : []),
    [nativeTierLane, executionTierModel, tierOptions],
  );

  const showTierPicker = useMemo(
    () =>
      nativeTierLane &&
      possibleTiers.length > 0 &&
      availableTiers.length > 0 &&
      (possibleTiers.length > 1 || scail2CandleTierLane),
    [nativeTierLane, possibleTiers, availableTiers, scail2CandleTierLane],
  );

  const baseExecutionTierBlockMessage =
    scail2CandleTierLane && availableTiers.length === 0
      ? "SCAIL-2 Candle generation requires an installed q4, q8, or bf16 tier. Install or repair one in Model Manager."
      : null;

  const hostMemory = useHostMemory();

  const nativeMemoryGb = hostMemoryGbForBackend(hostMemory, activeBackend);

  const autoTier = useMemo(
    () => suggestTier(executionTierModel, nativeMemoryGb, { backend: activeBackend }),
    [executionTierModel, nativeMemoryGb, activeBackend],
  );

  // Seed from the per-(video, model) sticky, then the global quality/Auto policy, clamped to installed.
  // A model transition always re-seeds even when both models happen to expose the same tier list.
  const {
    quantTier,
    setQuantTier,
    tierSwitching,
    handleTierChange,
  } = useQuantTierPicker({
    screen: TIER_SCREEN,
    model,
    selectedModel: executionTierModel,
    availableTiers,
    tierOptions,
    autoTier,
    useGenerationQuality: true,
    reseedOnModelChange: true,
    preferencesHydrated,
  });

  const availableTierKey = availableTiers.join(",");

  const recipeTierTargetModel = recipeTierRequest
    ? videoModels.find((item) => item.id === recipeTierRequest.modelId)
    : null;

  const recipeTierTargetAvailable = Boolean(
    recipeTierRequest && macVideoModels.some((item) => item.id === recipeTierRequest.modelId),
  );

  // Re-activate a replay target after a transient capability/catalog refresh when it can still
  // serve the current mode, then apply its exact tier. A user mode change that the target cannot
  // serve deliberately keeps the automatic fallback model active; the global replay guard below
  // still refuses submission until the user explicitly starts a new generation.
  useEffect(() => {
    if (
      recipeTierRequest &&
      selectedModel?.id !== recipeTierRequest.modelId &&
      recipeTierTargetAvailable &&
      recipeTierTargetModel &&
      videoModelServesMode(recipeTierTargetModel, mode, macCapabilities)
    ) {
      setModel(recipeTierRequest.modelId);
      return;
    }
    if (
      recipeTierRequest &&
      recipeTierTargetAvailable &&
      recipeTierRequest.modelId === selectedModel?.id &&
      availableTiers.includes(recipeTierRequest.tier)
    ) {
      setQuantTier(recipeTierRequest.tier);
    }
    // The stable key represents exact tier availability; the array identity changes with renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    recipeTierRequest,
    recipeTierTargetModel,
    recipeTierTargetAvailable,
    selectedModel?.id,
    mode,
    activeBackend,
    availableTierKey,
  ]);

  const handleExecutionTierChange = useCallback(
    (nextTier) => {
      setRecipeTierRequest(null);
      handleTierChange(nextTier);
    },
    [handleTierChange, setRecipeTierRequest],
  );

  const replayTierTargetsActiveModel = Boolean(
    recipeTierRequest && recipeTierRequest.modelId === selectedModel?.id,
  );

  const replayTierTargetName =
    recipeTierTargetModel?.name ?? recipeTierRequest?.modelName ?? recipeTierRequest?.modelId;

  const replayTierBlockMessage = recipeTierRequest
    ? !recipeTierTargetAvailable
      ? `This recipe requires ${replayTierTargetName} at ${tierLabel(recipeTierRequest.tier)}, but that model is not available on the current backend. Generation stays blocked while the recipe replay is active.`
      : !replayTierTargetsActiveModel
        ? `This recipe requires ${replayTierTargetName} at ${tierLabel(recipeTierRequest.tier)}, but ${selectedModel?.name ?? model} is active for the current mode. Generation stays blocked while the recipe replay is active.`
      : !availableTiers.includes(recipeTierRequest.tier)
      ? unavailableRecipeTierMessage(
          selectedModel,
          activeBackend,
          recipeTierRequest.tier,
        )
      : quantTier !== recipeTierRequest.tier
        ? `Preparing this recipe's exact ${tierLabel(recipeTierRequest.tier)} tier. Generation stays blocked until that tier is selected.`
        : null
    : null;

  const executionTierBlockMessage = replayTierBlockMessage ?? baseExecutionTierBlockMessage;

  const canStartNewBf16FromReplay =
    replayTierTargetsActiveModel &&
    scail2CandleTierLane &&
    recipeTierRequest.tier !== "bf16" &&
    !SCAIL2_CANDLE_PRODUCT_TIERS.includes(recipeTierRequest.tier) &&
    availableTiers.includes("bf16");

  const canStartNewGenerationFromReplay = Boolean(
    replayTierBlockMessage &&
      model === selectedModel?.id &&
      macVideoModels.some((item) => item.id === model) &&
      !baseExecutionTierBlockMessage &&
      (!nativeTierLane || availableTiers.includes(quantTier)),
  );

  const startNewGenerationLabel = canStartNewBf16FromReplay
    ? "Use bf16 for a new generation"
    : `Start new generation with ${selectedModel?.name ?? model}${
        nativeTierLane ? ` ${tierLabel(quantTier)}` : ""
      }`;

  const handleStartNewGenerationFromReplay = useCallback(() => {
    // Recheck the same identity/tier contract at the click boundary. During a catalog refresh,
    // `selectedModel` can already derive the fallback while `model` still carries the removed id;
    // clearing replay in that transient render would advertise one model and submit another.
    if (
      !recipeTierRequest ||
      !replayTierBlockMessage ||
      model !== selectedModel?.id ||
      !macVideoModels.some((item) => item.id === model) ||
      baseExecutionTierBlockMessage ||
      (nativeTierLane && !availableTiers.includes(quantTier))
    ) {
      return;
    }
    setRecipeModelNotice("");
    setRecipeTierRequest(null);
    if (nativeTierLane && availableTiers.includes(quantTier)) {
      handleTierChange(quantTier);
    }
  }, [recipeTierRequest, replayTierBlockMessage, model, selectedModel?.id, macVideoModels, baseExecutionTierBlockMessage, nativeTierLane, availableTiers, quantTier, setRecipeModelNotice, setRecipeTierRequest, handleTierChange]);

  const showTorchQuantization = activeBackend !== "mlx" && !nativeTierLane && supportsQuantization;

  const selectedTierQuantize =
    nativeTierLane && availableTiers.includes(quantTier) ? tierQuantize(quantTier) : null;

  const tierHasMemoryRisk = showTierPicker && ["q8", "bf16"].includes(quantTier);
  return { activeBackend, tierPickerItems, showTierPicker, quantTier, tierSwitching, handleExecutionTierChange, executionTierBlockMessage, canStartNewGenerationFromReplay, startNewGenerationLabel, handleStartNewGenerationFromReplay, showTorchQuantization, selectedTierQuantize, tierHasMemoryRisk };
}
