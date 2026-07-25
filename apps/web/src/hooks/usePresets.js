import { useScopedCrudList } from "./useScopedCrudList.js";

export function usePresets(options) {
  const catalog = useScopedCrudList({
    ...options,
    resourcePath: "/api/v1/recipe-presets",
    projectRequiredMessage: "Create or open a project first.",
  });
  return {
    presets: catalog.items,
    setPresets: catalog.setItems,
    refreshPresets: catalog.refresh,
    createPreset: catalog.create,
    updatePreset: catalog.update,
    duplicatePreset: catalog.duplicate,
    deletePreset: catalog.remove,
  };
}
