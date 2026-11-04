import { useAppDomainValue } from "../context/useAppDomainValue.js";
import { recipePresetsRequest } from "../api/scopedCatalogs.js";
import { useScopedCrudList } from "./useScopedCrudList.js";

export function usePresets(options) {
  const catalog = useScopedCrudList({
    ...options,
    request: recipePresetsRequest,
    projectRequiredMessage: "Create or open a project first.",
  });
  const context = useAppDomainValue({
    presets: catalog.items,
    createPreset: catalog.create,
    updatePreset: catalog.update,
    duplicatePreset: catalog.duplicate,
    deletePreset: catalog.remove
  });
  return { ...context, setPresets: catalog.setItems, refreshPresets: catalog.refresh, context };
}
