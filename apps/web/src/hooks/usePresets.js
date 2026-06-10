import { useCallback, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";

// Owns the recipe-preset list plus its scoped refresh/create/update/duplicate/delete
// mutations. Extracted from App.jsx (sc-1651). Behavior unchanged — token, the active
// project, and error reporting are passed in. App's bulk refreshData still seeds the
// list via the returned setPresets (same React setter identity), and the project-load
// effect calls refreshCharacters/refreshPresets exactly as before.
export function usePresets({ token, activeProject, setError }) {
  const [presets, setPresets] = useState([]);

  // sc-4194: actions wrapped in useCallback so their identity is stable across
  // App's SSE-driven re-renders, enabling appContextValue to memoize.
  const refreshPresets = useCallback(
    async (projectId = activeProject?.id, { signal } = {}) => {
      try {
        const query = projectId ? `?projectId=${encodeURIComponent(projectId)}` : "";
        const items = await apiFetch(`/api/v1/recipe-presets${query}`, token, { signal });
        setPresets(items);
        setError("");
        return items;
      } catch (err) {
        if (isAbortError(err)) return [];
        setError(err.message);
        return [];
      }
    },
    [token, activeProject, setError],
  );

  const presetQuery = useCallback(
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

  const createPreset = useCallback(
    async (payload) => {
      if (payload.scope === "project" && !activeProject) {
        throw new Error("Create or open a project first.");
      }
      const created = await apiFetch(`/api/v1/recipe-presets${presetQuery(payload.scope)}`, token, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      await refreshPresets(activeProject?.id);
      return created;
    },
    [token, activeProject, presetQuery, refreshPresets],
  );

  const updatePreset = useCallback(
    async (presetId, payload, scope = payload.scope) => {
      const updated = await apiFetch(`/api/v1/recipe-presets/${encodeURIComponent(presetId)}${presetQuery(scope)}`, token, {
        method: "PATCH",
        body: JSON.stringify(payload),
      });
      await refreshPresets(activeProject?.id);
      return updated;
    },
    [token, activeProject, presetQuery, refreshPresets],
  );

  const duplicatePreset = useCallback(
    async (presetId, scope = null) => {
      const duplicated = await apiFetch(`/api/v1/recipe-presets/${encodeURIComponent(presetId)}/duplicate${presetQuery(scope)}`, token, {
        method: "POST",
        body: JSON.stringify({}),
      });
      await refreshPresets(activeProject?.id);
      return duplicated;
    },
    [token, activeProject, presetQuery, refreshPresets],
  );

  const deletePreset = useCallback(
    async (presetId, scope = null) => {
      const archived = await apiFetch(`/api/v1/recipe-presets/${encodeURIComponent(presetId)}${presetQuery(scope)}`, token, {
        method: "DELETE",
      });
      await refreshPresets(activeProject?.id);
      return archived;
    },
    [token, activeProject, presetQuery, refreshPresets],
  );

  return {
    presets,
    setPresets,
    refreshPresets,
    createPreset,
    updatePreset,
    duplicatePreset,
    deletePreset,
  };
}
