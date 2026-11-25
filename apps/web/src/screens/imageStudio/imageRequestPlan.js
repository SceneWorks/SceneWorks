import { buildImageJobRequest, composeImageJobPrompt } from "../../imageJobRequest.js";
import { fitPromptToLimit } from "../../imagePromptLimit.js";
import { composePreset } from "../../presetUtils.js";
import { parseMagicPromptCaption } from "../../ideogramCaption.js";
import { H3_IMAGE_EXPAND_COPY, h3ImageProposedResolution } from "../../minimaxH3ImageExpand.js";
import { isModelMissing } from "../../modelMissing.js";

// Factories deliberately live outside the component. A background batch keeps
// only its request inputs, never a closure over the studio's React fiber/media.
export function createPromptComposer({
  generalStack, selectedPreset, resolutionOptions, mossEdit, mossPromptBatchCount,
  activeStyleText, appearance, selectedModel,
}) {
  return (overrides = {}) => {
    const isStructured = overrides.sendStructured ?? false;
    const foldPrompt = (generalStack.length > 0 || (mossEdit && mossPromptBatchCount > 0)) && !isStructured;
    const promptToSend = foldPrompt && overrides.promptToSend != null
      ? composePreset({ base: selectedPreset, generalStack, userText: overrides.promptToSend, resolutionOptions }).prompt
      : overrides.promptToSend;
    const { prompt: composedPrompt } = composeImageJobPrompt({
      promptToSend, sendStructured: isStructured, styleText: activeStyleText, appearance, model: selectedModel,
    });
    return { composedPrompt, foldPrompt, isStructured, promptToSend };
  };
}

export function createImageRequestBuilder({
  settings, composeJobPrompt, mossEdit, mossPromptBatch, stackResolution,
  promptCharLimit, caption, magicPromptBackend,
}) {
  const buildJobRequest = (overrides = {}) => {
    const { foldPrompt, isStructured, promptToSend } = composeJobPrompt(overrides);
    return buildImageJobRequest({
      ...settings,
      promptToSend,
      appearancePromptBatch: overrides.appearancePromptBatch?.map((text) => composeJobPrompt({ ...overrides, promptToSend: text }).promptToSend),
      submitIntent: overrides.submitIntent,
      sendStructured: isStructured,
      submitCaption: overrides.submitCaption,
      submitBackend: overrides.submitBackend,
      resolutionOverride: overrides.resolutionOverride ?? stackResolution,
      mossPromptBatch: mossEdit
        ? mossPromptBatch.map((text) => composeJobPrompt({ ...overrides, promptToSend: text }).promptToSend)
        : mossPromptBatch,
      presetPromptResolvedClientSide: foldPrompt,
    });
  };
  const fit = (text) => fitPromptToLimit(text, promptCharLimit,
    (candidate) => composeJobPrompt({ promptToSend: candidate }).composedPrompt).prompt;
  const buildBatchJobRequest = (resolvedPrompt, opts = {}) => {
    const fitted = opts.promptToSend == null ? fit(resolvedPrompt) : resolvedPrompt;
    return buildJobRequest({
      promptToSend: opts.promptToSend ?? fitted,
      appearancePromptBatch: opts.appearancePromptBatch?.map(fit),
      submitIntent: fitted,
      sendStructured: opts.sendStructured ?? false,
      submitCaption: opts.submitCaption ?? caption,
      submitBackend: opts.submitBackend ?? magicPromptBackend,
      resolutionOverride: opts.resolution ?? null,
    });
  };
  return { buildJobRequest, buildBatchJobRequest };
}

export function createBatchMagicExpand({ magicPrompt, width, height, model }) {
  const gcd = (a, b) => b ? gcd(b, a % b) : a;
  const divisor = gcd(width, height) || 1;
  const aspectRatio = Number.isFinite(width) && Number.isFinite(height) ? `${width / divisor}:${height / divisor}` : "1:1";
  return async (idea) => {
    if (typeof magicPrompt !== "function") throw new Error("Magic-prompt is unavailable.");
    const raw = await magicPrompt({ prompt: idea, modelId: model, aspectRatio });
    const { caption, error } = parseMagicPromptCaption(raw);
    if (error || !caption) throw new Error(error || "Magic-prompt returned an unusable caption.");
    return caption;
  };
}

export function createBatchExpansion({
  expandPrompt, expanderModel, modelId, referenceAssetIds, useSizes, selectedModel, resolution,
}) {
  return {
    expandPrompt: (args) => expandPrompt({ ...args, expanderModelId: expanderModel?.id }),
    copy: H3_IMAGE_EXPAND_COPY,
    modelId, referenceAssetIds, useSizes,
    proposedResolution: (suggestion) => h3ImageProposedResolution(suggestion?.whRatio, selectedModel, resolution),
    failureMessage: (error) => isModelMissing(expanderModel, error?.message)
      ? H3_IMAGE_EXPAND_COPY.missing(false, expanderModel?.name || H3_IMAGE_EXPAND_COPY.defaultModelName)
      : error?.message || H3_IMAGE_EXPAND_COPY.failure,
  };
}
