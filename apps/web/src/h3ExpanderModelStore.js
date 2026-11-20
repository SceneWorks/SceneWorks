// The MiniMax-H3-Image expander's saved model (SW-7208): the localStorage cache of the durable
// server preference. Kept apart from minimaxH3ImageExpand.js because App.jsx seeds it on launch, and
// App's static imports are the initial bundle: the studios' expander UI and wording stay in their
// own lazy chunks.

import { MINIMAX_H3_IMAGE_PE_MODEL_ID } from "./constants.js";
import { putUiPreferences } from "./uiPreferences.js";

// The models the expander can run. Mirrors `sceneworks_core::contracts::MINIMAX_H3_IMAGE_EXPANDER_MODEL_IDS`.
export const H3_EXPANDER_DESCRIBER_MODEL_ID = "vision_describe_qwen3_8_27b";
export const H3_EXPANDER_MODEL_IDS = Object.freeze([MINIMAX_H3_IMAGE_PE_MODEL_ID, H3_EXPANDER_DESCRIBER_MODEL_ID]);

const STORAGE_KEY = "sceneworks-h3-image-expander-model";

export function normalizeH3ExpanderModel(value) {
  return H3_EXPANDER_MODEL_IDS.includes(value) ? value : MINIMAX_H3_IMAGE_PE_MODEL_ID;
}

// The saved expander model: the localStorage cache of the durable server preference.
export function readH3ExpanderModel() {
  try {
    return normalizeH3ExpanderModel(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return MINIMAX_H3_IMAGE_PE_MODEL_ID;
  }
}

function cacheH3ExpanderModel(value) {
  try {
    window.localStorage.setItem(STORAGE_KEY, value);
  } catch {
    // localStorage unavailable: the server copy still holds the choice.
  }
}

// Seed the cache from the durable server copy on launch (App.jsx, after GET /api/v1/ui-preferences).
export function seedH3ExpanderModelFromServer(value) {
  if (H3_EXPANDER_MODEL_IDS.includes(value)) cacheH3ExpanderModel(value);
}

// Save the user's pick: the instant-paint cache and the durable server preference, which the API
// also reads for an expansion request that names no model.
export function writeH3ExpanderModel(value) {
  const model = normalizeH3ExpanderModel(value);
  cacheH3ExpanderModel(model);
  putUiPreferences({ h3ImageExpanderModel: model }).catch(() => {});
  return model;
}
