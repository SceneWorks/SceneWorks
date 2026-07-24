import { useCallback, useMemo, useRef, useState } from "react";

import { cardinality, extractKeys, splitPromptLines } from "../../promptBatch.js";

function batchTextFromPrompts(prompts) {
  const list = Array.isArray(prompts) ? prompts : [];
  return list.join(list.some((prompt) => prompt.includes("\n")) ? "\n---\n" : "\n");
}

export function useBatchPromptState({
  saved,
  createPromptBatch,
  updatePromptBatch,
  deletePromptBatch,
}) {
  const [batchMode, setBatchMode] = useState(saved.batchMode ?? false);
  const [batchPromptsText, setBatchPromptsText] = useState(saved.batchPromptsText ?? "");
  const [batchVariableValues, setBatchVariableValues] = useState(saved.batchVariableValues ?? {});
  const [batchName, setBatchName] = useState(saved.batchName ?? "");
  const [batchScope, setBatchScope] = useState(saved.batchScope ?? "global");
  const [loadedBatchId, setLoadedBatchId] = useState(saved.loadedBatchId ?? null);
  const [batchError, setBatchError] = useState("");
  const [batchBusy, setBatchBusy] = useState(false);
  const [batchRun, setBatchRun] = useState(null);
  const [batchConfirmPending, setBatchConfirmPending] = useState(false);
  const batchAbortRef = useRef(false);

  const batchPrompts = useMemo(() => splitPromptLines(batchPromptsText), [batchPromptsText]);
  const batchVariables = useMemo(
    () =>
      extractKeys(batchPrompts).map((key) => ({
        key,
        values: (batchVariableValues[key] ?? []).filter((value) => value.trim() !== ""),
      })),
    [batchPrompts, batchVariableValues],
  );
  const batchJobCount = useMemo(
    () => cardinality(batchPrompts, batchVariables, 1),
    [batchPrompts, batchVariables],
  );

  const applyBatchContent = useCallback(({ prompts, variables, lastValues, name }) => {
    setBatchPromptsText(batchTextFromPrompts(prompts));
    const values = {};
    for (const variable of variables ?? []) {
      if (variable?.key) values[variable.key] = Array.isArray(variable.values) ? variable.values : [];
    }
    for (const [key, vals] of Object.entries(lastValues ?? {})) {
      if (!(key in values) && Array.isArray(vals)) values[key] = vals;
    }
    setBatchVariableValues(values);
    if (name !== undefined) setBatchName(name ?? "");
    setBatchError("");
  }, []);

  const handleSaveBatch = useCallback(async () => {
    setBatchBusy(true);
    setBatchError("");
    try {
      const payload = {
        name: batchName.trim(),
        scope: batchScope,
        prompts: batchPrompts,
        variables: batchVariables,
        lastValues: Object.fromEntries(batchVariables.map((variable) => [variable.key, variable.values])),
      };
      const result = loadedBatchId
        ? await updatePromptBatch(loadedBatchId, payload, batchScope)
        : await createPromptBatch(payload);
      if (result?.id) setLoadedBatchId(result.id);
    } catch (error) {
      setBatchError(error.message);
    } finally {
      setBatchBusy(false);
    }
  }, [batchName, batchScope, batchPrompts, batchVariables, loadedBatchId, updatePromptBatch, createPromptBatch]);

  const handleLoadBatch = useCallback((batch) => {
    applyBatchContent(batch);
    setBatchScope(batch.scope === "project" ? "project" : "global");
    setLoadedBatchId(batch.id ?? null);
  }, [applyBatchContent]);
  const handleDeleteBatch = useCallback(async (batch) => {
    setBatchError("");
    try {
      await deletePromptBatch(batch.id, batch.scope);
      setLoadedBatchId((current) => (current === batch.id ? null : current));
    } catch (error) {
      setBatchError(error.message);
    }
  }, [deletePromptBatch]);
  const handleImportBatch = useCallback((payload) => {
    applyBatchContent(payload);
    setLoadedBatchId(null);
  }, [applyBatchContent]);
  const handleNewBatch = useCallback(() => {
    setBatchPromptsText("");
    setBatchVariableValues({});
    setBatchName("");
    setLoadedBatchId(null);
    setBatchError("");
  }, []);

  return {
    batchMode, setBatchMode, batchPromptsText, setBatchPromptsText,
    batchVariableValues, setBatchVariableValues, batchName, setBatchName,
    batchScope, setBatchScope, loadedBatchId, setLoadedBatchId, batchError,
    setBatchError, batchBusy, batchRun, setBatchRun, batchConfirmPending,
    setBatchConfirmPending, batchAbortRef, batchPrompts, batchVariables,
    batchJobCount, handleSaveBatch, handleLoadBatch, handleDeleteBatch,
    handleImportBatch, handleNewBatch,
  };
}
