import { useCallback, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { isCurrentProjectRequest } from "../appStateHelpers.js";

// Owns the prompt-batch list plus its scoped refresh/create/update/duplicate/delete
// mutations (sc-9954, epic 9952). A direct sibling of usePresets — same scoping and
// error handling — against the /api/v1/prompt-batches routes. Batches carry prompt
// templates + variable definitions, not a generation recipe, so there is no
// model/workflow filtering here.
export function usePromptBatches({ token, activeProject, activeProjectRef, setError }) {
  const [promptBatches, setPromptBatches] = useState([]);

  const isCurrentBatchRequest = useCallback(
    (projectId) =>
      !projectId ||
      isCurrentProjectRequest(activeProjectRef?.current?.id ?? null, projectId),
    [activeProjectRef],
  );

  const refreshPromptBatches = useCallback(
    async (projectId = activeProject?.id, { signal } = {}) => {
      try {
        const query = projectId ? `?projectId=${encodeURIComponent(projectId)}` : "";
        const items = await apiFetch(`/api/v1/prompt-batches${query}`, token, { signal });
        if (!isCurrentBatchRequest(projectId)) return [];
        setPromptBatches(items);
        setError("");
        return items;
      } catch (err) {
        if (isAbortError(err)) return [];
        if (!isCurrentBatchRequest(projectId)) return [];
        setError(err.message);
        return [];
      }
    },
    [token, activeProject, isCurrentBatchRequest, setError],
  );

  const batchQuery = useCallback(
    (scope = null) => {
      const params = new URLSearchParams();
      if (scope) {
        params.set("scope", scope);
      }
      if (scope === "project" && activeProject?.id) {
        params.set("projectId", activeProject.id);
      }
      const value = params.toString();
      return value ? `?${value}` : "";
    },
    [activeProject],
  );

  const createPromptBatch = useCallback(
    async (payload) => {
      if (payload.scope === "project" && !activeProject) {
        throw new Error("Create or open a project first.");
      }
      const created = await apiFetch(`/api/v1/prompt-batches${batchQuery(payload.scope)}`, token, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      // Refresh the caller's current catalog view even when the mutation itself
      // targets a global batch: a project view contains both global and project
      // batches. Capture the id so a later project switch can suppress this refetch.
      const projectId = activeProject?.id ?? null;
      if (isCurrentBatchRequest(projectId)) {
        await refreshPromptBatches(projectId);
      }
      return created;
    },
    [token, activeProject, batchQuery, isCurrentBatchRequest, refreshPromptBatches],
  );

  const updatePromptBatch = useCallback(
    async (batchId, payload, scope = payload.scope) => {
      const updated = await apiFetch(
        `/api/v1/prompt-batches/${encodeURIComponent(batchId)}${batchQuery(scope)}`,
        token,
        { method: "PATCH", body: JSON.stringify(payload) },
      );
      const projectId = activeProject?.id ?? null;
      if (isCurrentBatchRequest(projectId)) {
        await refreshPromptBatches(projectId);
      }
      return updated;
    },
    [token, activeProject, batchQuery, isCurrentBatchRequest, refreshPromptBatches],
  );

  const duplicatePromptBatch = useCallback(
    async (batchId, scope = null) => {
      const duplicated = await apiFetch(
        `/api/v1/prompt-batches/${encodeURIComponent(batchId)}/duplicate${batchQuery(scope)}`,
        token,
        { method: "POST", body: JSON.stringify({}) },
      );
      const projectId = activeProject?.id ?? null;
      if (isCurrentBatchRequest(projectId)) {
        await refreshPromptBatches(projectId);
      }
      return duplicated;
    },
    [token, activeProject, batchQuery, isCurrentBatchRequest, refreshPromptBatches],
  );

  const deletePromptBatch = useCallback(
    async (batchId, scope = null) => {
      const archived = await apiFetch(
        `/api/v1/prompt-batches/${encodeURIComponent(batchId)}${batchQuery(scope)}`,
        token,
        { method: "DELETE" },
      );
      const projectId = activeProject?.id ?? null;
      if (isCurrentBatchRequest(projectId)) {
        await refreshPromptBatches(projectId);
      }
      return archived;
    },
    [token, activeProject, batchQuery, isCurrentBatchRequest, refreshPromptBatches],
  );

  return {
    promptBatches,
    setPromptBatches,
    refreshPromptBatches,
    createPromptBatch,
    updatePromptBatch,
    duplicatePromptBatch,
    deletePromptBatch,
  };
}
