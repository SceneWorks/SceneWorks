import { useCallback, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { isCurrentProjectRequest } from "../appStateHelpers.js";
import { refreshFailure, refreshSuccess } from "../refreshResult.js";

// Shared global/project-scoped catalog mutations. Recipe presets and prompt
// batches intentionally expose different domain names, but their request,
// stale-project suppression, refresh and error behavior are one contract.
export function useScopedCrudList({
  token,
  activeProject,
  activeProjectRef,
  setError,
  resourcePath,
  projectRequiredMessage,
}) {
  const [items, setItems] = useState([]);

  const isCurrentRequest = useCallback(
    (projectId) =>
      !projectId ||
      isCurrentProjectRequest(activeProjectRef?.current?.id ?? null, projectId),
    [activeProjectRef],
  );

  const refresh = useCallback(
    async (projectId = activeProject?.id, { signal } = {}) => {
      if (
        projectId &&
        !isCurrentProjectRequest(activeProject?.id ?? null, projectId)
      ) return refreshFailure("stale");
      try {
        const query = projectId ? `?projectId=${encodeURIComponent(projectId)}` : "";
        const next = await apiFetch(`${resourcePath}${query}`, token, { signal });
        if (!isCurrentRequest(projectId)) return refreshFailure("stale");
        setItems(next);
        setError("");
        return refreshSuccess(next);
      } catch (error) {
        if (!isCurrentRequest(projectId)) return refreshFailure("stale", error);
        if (isAbortError(error)) return refreshFailure("aborted", error);
        setError(error.message);
        return refreshFailure("error", error);
      }
    },
    [activeProject, isCurrentRequest, resourcePath, setError, token],
  );

  const scopeQuery = useCallback(
    (scope = null) => {
      const params = new URLSearchParams();
      if (scope) params.set("scope", scope);
      if (scope === "project" && activeProject?.id) {
        params.set("projectId", activeProject.id);
      }
      const value = params.toString();
      return value ? `?${value}` : "";
    },
    [activeProject],
  );

  const refreshCurrent = useCallback(async () => {
    const projectId = activeProject?.id ?? null;
    if (isCurrentRequest(projectId)) await refresh(projectId);
  }, [activeProject, isCurrentRequest, refresh]);

  const create = useCallback(
    async (payload) => {
      if (payload.scope === "project" && !activeProject) {
        throw new Error(projectRequiredMessage);
      }
      const created = await apiFetch(`${resourcePath}${scopeQuery(payload.scope)}`, token, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      await refreshCurrent();
      return created;
    },
    [activeProject, projectRequiredMessage, refreshCurrent, resourcePath, scopeQuery, token],
  );

  const update = useCallback(
    async (id, payload, scope = payload.scope) => {
      const updated = await apiFetch(
        `${resourcePath}/${encodeURIComponent(id)}${scopeQuery(scope)}`,
        token,
        { method: "PATCH", body: JSON.stringify(payload) },
      );
      await refreshCurrent();
      return updated;
    },
    [refreshCurrent, resourcePath, scopeQuery, token],
  );

  const duplicate = useCallback(
    async (id, scope = null) => {
      const duplicated = await apiFetch(
        `${resourcePath}/${encodeURIComponent(id)}/duplicate${scopeQuery(scope)}`,
        token,
        { method: "POST", body: JSON.stringify({}) },
      );
      await refreshCurrent();
      return duplicated;
    },
    [refreshCurrent, resourcePath, scopeQuery, token],
  );

  const remove = useCallback(
    async (id, scope = null) => {
      const removed = await apiFetch(
        `${resourcePath}/${encodeURIComponent(id)}${scopeQuery(scope)}`,
        token,
        { method: "DELETE" },
      );
      await refreshCurrent();
      return removed;
    },
    [refreshCurrent, resourcePath, scopeQuery, token],
  );

  return { items, setItems, refresh, create, update, duplicate, remove };
}
