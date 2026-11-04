import { useAppDomainValue } from "../context/useAppDomainValue.js";
import { promptBatchesRequest } from "../api/scopedCatalogs.js";
import { useScopedCrudList } from "./useScopedCrudList.js";

export function usePromptBatches(options) {
  const catalog = useScopedCrudList({
    ...options,
    request: promptBatchesRequest,
    projectRequiredMessage: "Create or open a project first.",
  });
  const context = useAppDomainValue({
    promptBatches: catalog.items,
    createPromptBatch: catalog.create,
    updatePromptBatch: catalog.update,
    duplicatePromptBatch: catalog.duplicate,
    deletePromptBatch: catalog.remove
  });
  return { ...context, setPromptBatches: catalog.setItems, refreshPromptBatches: catalog.refresh, context };
}
