import { useCallback, useMemo, useState } from "react";
import { useAppContext } from "../../context/AppContext.js";

// Which model "Make a picture" uses. Simple mode previously hard-wired
// imageModels[0] — whatever the manifest happened to list first (a fast/weak
// turbo) — with no way to change it. This exposes a friendly picker and a
// sensible default, persisted per browser.
export const SIMPLE_IMAGE_MODEL_KEY = "sceneworks-simple-image-model";

// Default preference for a general creative surface: SDXL is the most versatile
// across every look (photo/anime/illustration/3D/watercolor); the photoreal and
// high-fidelity options come next; the turbo model is the last resort, not the
// default. First installed wins.
const IMAGE_MODEL_PREFERENCE = ["sdxl", "realvisxl", "sensenova_u1_8b", "sensenova_u1_8b_fast", "z_image_turbo"];

// Only genuine text-to-image models belong in the picker — not the edit-only or
// identity/reference (character) models, which can't run a plain prompt.
export function textToImageModels(imageModels = []) {
  return imageModels.filter((model) => (model.capabilities ?? []).includes("text_to_image"));
}

export function defaultImageModelId(models = []) {
  for (const id of IMAGE_MODEL_PREFERENCE) {
    if (models.some((model) => model.id === id)) return id;
  }
  return models[0]?.id ?? null;
}

export function readSimpleImageModel() {
  if (typeof localStorage === "undefined") return null;
  try {
    return localStorage.getItem(SIMPLE_IMAGE_MODEL_KEY) || null;
  } catch {
    return null;
  }
}

export function writeSimpleImageModel(id) {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(SIMPLE_IMAGE_MODEL_KEY, id);
  } catch {
    // Private mode — the picker still works for the session, just doesn't persist.
  }
}

export function modelLabel(model) {
  return model?.ui?.label ?? model?.name ?? model?.id ?? "Model";
}

// The resolved text-to-image model for Simple mode: the user's saved choice if
// it's still installed, otherwise the preference-ordered default. Shared by Make
// a picture and the look-exemplar previews so previews match what Create renders.
export function useSimpleImageModel() {
  const { imageModels = [] } = useAppContext();
  const models = useMemo(() => textToImageModels(imageModels), [imageModels]);
  const [chosenId, setChosenId] = useState(() => readSimpleImageModel());

  const modelId = useMemo(() => {
    if (chosenId && models.some((model) => model.id === chosenId)) return chosenId;
    return defaultImageModelId(models);
  }, [chosenId, models]);

  const model = useMemo(() => models.find((entry) => entry.id === modelId) ?? null, [models, modelId]);
  const [savedDefault, setSavedDefault] = useState(() => readSimpleImageModel());

  // Session-only: changing the picker doesn't persist until "Make my default".
  const select = useCallback((id) => setChosenId(id), []);
  const makeDefault = useCallback(() => {
    if (!modelId) return;
    writeSimpleImageModel(modelId);
    setSavedDefault(modelId);
  }, [modelId]);
  const isDefault = Boolean(modelId) && modelId === savedDefault;

  return { models, model, modelId, select, makeDefault, isDefault };
}
