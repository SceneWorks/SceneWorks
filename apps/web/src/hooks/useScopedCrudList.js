import { useCallback, useState } from "react";
import { useProjectScopedList } from "./useProjectScopedList.js";
import { isCurrentProjectRequest } from "../appStateHelpers.js";

// Shared global/project-scoped catalog mutations. Recipe presets and prompt
// batches intentionally expose different domain names, but their request,
// stale-project suppression, refresh and error behavior are one contract.
export function useScopedCrudList({
  token,
  activeProject,
  activeProjectRef,
  setError,
  request,
  projectRequiredMessage,
}) {
  const [items, setItems] = useState([]);

  const isCurrentRequest = useCallback(
    (projectId) =>
      !projectId ||
      isCurrentProjectRequest(activeProjectRef?.current?.id ?? null, projectId),
    [activeProjectRef],
  );

  const refresh = useProjectScopedList({
    token, activeProject, activeProjectRef, setError, optionalProject: true,
    load: (projectId, token, options) => request(null, token, options, { projectId }),
    setItems,
  });

  const scopeOptions = useCallback((scope = null) => ({
    scope,
    projectId: scope === "project" ? activeProject?.id : undefined,
  }), [activeProject]);

  const refreshCurrent = useCallback(async () => {
    const projectId = activeProject?.id ?? null;
    if (isCurrentRequest(projectId)) await refresh(projectId);
  }, [activeProject, isCurrentRequest, refresh]);

  const create = useCallback(
    async (payload) => {
      if (payload.scope === "project" && !activeProject) {
        throw new Error(projectRequiredMessage);
      }
      const created = await request(null, token, {
        method: "POST",
        body: JSON.stringify(payload),
      }, scopeOptions(payload.scope));
      await refreshCurrent();
      return created;
    },
    [activeProject, projectRequiredMessage, refreshCurrent, request, scopeOptions, token],
  );

  const update = useCallback(
    async (id, payload, scope = payload.scope) => {
      const updated = await request(
        id,
        token,
        { method: "PATCH", body: JSON.stringify(payload) }, scopeOptions(scope),
      );
      await refreshCurrent();
      return updated;
    },
    [refreshCurrent, request, scopeOptions, token],
  );

  const duplicate = useCallback(
    async (id, scope = null) => {
      const duplicated = await request(
        id,
        token,
        { method: "POST", body: JSON.stringify({}) }, { ...scopeOptions(scope), duplicate: true },
      );
      await refreshCurrent();
      return duplicated;
    },
    [refreshCurrent, request, scopeOptions, token],
  );

  const remove = useCallback(
    async (id, scope = null) => {
      const removed = await request(
        id,
        token,
        { method: "DELETE" }, scopeOptions(scope),
      );
      await refreshCurrent();
      return removed;
    },
    [refreshCurrent, request, scopeOptions, token],
  );

  return { items, setItems, refresh, create, update, duplicate, remove };
}
