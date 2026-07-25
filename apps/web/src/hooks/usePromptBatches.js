import { useScopedCrudList } from "./useScopedCrudList.js";

export function usePromptBatches(options) {
  const catalog = useScopedCrudList({
    ...options,
    resourcePath: "/api/v1/prompt-batches",
    projectRequiredMessage: "Create or open a project first.",
  });
  return {
    promptBatches: catalog.items,
    setPromptBatches: catalog.setItems,
    refreshPromptBatches: catalog.refresh,
    createPromptBatch: catalog.create,
    updatePromptBatch: catalog.update,
    duplicatePromptBatch: catalog.duplicate,
    deletePromptBatch: catalog.remove,
  };
}
