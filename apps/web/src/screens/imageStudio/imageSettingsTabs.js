import { nearestAspectRatio } from "../../aspectResolution.js";
import { findStyleById } from "../../data/styleCatalog.js";

// The Image Studio settings tab strip (studio layout simplification): ids, the persisted-state
// migration, and the one-line summaries each tab shows under its label.

export const IMAGE_SETTINGS_TABS = ["model", "loras", "style", "sampling", "finish"];
export const DEFAULT_IMAGE_SETTINGS_TAB = "model";

// The open tab from a saved workspace snapshot. `settingsTab` wins when present (null = the user
// closed every tab). A snapshot from before the tab strip has only `advancedOpen`: an open
// Advanced section reopens as Sampling, the tab that holds most of what Advanced held.
export function imageSettingsTabFromSnapshot(saved) {
  if (saved && Object.hasOwn(saved, "settingsTab")) {
    const tab = saved.settingsTab;
    if (tab === null) return null;
    return IMAGE_SETTINGS_TABS.includes(tab) ? tab : DEFAULT_IMAGE_SETTINGS_TAB;
  }
  return saved?.advancedOpen === true ? "sampling" : DEFAULT_IMAGE_SETTINGS_TAB;
}

const filled = (value) => value !== "" && value != null;

export function imageSettingsTabSummaries({
  selectedModel,
  width,
  height,
  sizeEnvelope,
  count,
  showTierPicker,
  quantTier,
  userSelectedLoraCount,
  styleAxisAvailable,
  styleId,
  selectedPreset,
  generalStackCount,
  appearanceActive,
  mossDmd,
  mossDmdSteps,
  stepsOverride,
  guidanceOverride,
  seed,
  upscaleEnabled,
  upscaleFactor,
  hiresFixEnabled,
  hiresFixUpscaleBy,
  usePid,
  pidTarget,
  decoder,
  transparentBackground,
}) {
  const ratio = width > 0 && height > 0
    ? nearestAspectRatio(width, height, sizeEnvelope?.availableRatios?.length ? sizeEnvelope.availableRatios : undefined)
    : null;
  const model = [
    selectedModel?.name ?? selectedModel?.id,
    ratio?.id,
    count ? `×${count}` : null,
    showTierPicker && quantTier ? quantTier : null,
  ].filter(Boolean).join(" · ");

  const loras = userSelectedLoraCount > 0 ? `${userSelectedLoraCount} selected` : "None";

  const styleName = styleAxisAvailable && styleId ? findStyleById(styleId)?.name ?? null : null;
  const styleParts = [
    styleName,
    selectedPreset ? selectedPreset.name ?? selectedPreset.id : null,
    appearanceActive ? "Look" : null,
    generalStackCount > 0 ? `+${generalStackCount} general` : null,
  ].filter(Boolean);
  const style = styleParts.length ? styleParts.join(" · ") : "None";

  const steps = mossDmd ? (filled(mossDmdSteps) ? mossDmdSteps : 8) : stepsOverride;
  const samplingParts = [
    filled(steps) ? `${steps} steps` : null,
    filled(guidanceOverride) && !mossDmd ? `cfg ${guidanceOverride}` : null,
    filled(seed) ? `seed ${seed}` : null,
  ].filter(Boolean);
  const sampling = samplingParts.length ? samplingParts.join(" · ") : "Model defaults";

  const finishParts = [
    upscaleEnabled && !usePid ? `Upscale ${upscaleFactor}x` : null,
    hiresFixEnabled ? `Hires ${hiresFixUpscaleBy}x` : null,
    usePid ? `PiD ${String(pidTarget ?? "").toUpperCase()}` : null,
    decoder && decoder !== "native" ? "Alt decoder" : null,
    transparentBackground ? "RGBA" : null,
  ].filter(Boolean);
  const finish = finishParts.length ? finishParts.join(" · ") : "Off";

  return {
    model,
    loras,
    lorasChanged: userSelectedLoraCount > 0,
    style,
    styleChanged: styleParts.length > 0,
    sampling,
    samplingChanged: samplingParts.length > 0,
    finish,
    finishChanged: finishParts.length > 0,
  };
}
