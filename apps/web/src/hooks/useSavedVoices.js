import { useCallback, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";

// Owns the project's Voice Clone "saved voices" roster + its create/delete mutations (sc-13517).
// A saved voice is a named pointer to a library audio reference clip (plus its Chatterbox-VE speaker
// embedding, computed + dedup-checked on the backend at register time). Extracted as a thin data
// layer like useCharacters/usePresets; shared concerns (token, active project, error) are passed in.
//
// Every returned action is useCallback-wrapped with stable deps so appStaticValue can stay memoized
// across App's SSE-driven re-renders (mirrors the sc-4194 note in useCharacters/useModelsAndLoras).
export function useSavedVoices({ token, activeProject, activeProjectRef, setError }) {
  const [savedVoices, setSavedVoices] = useState([]);

  const refreshSavedVoices = useCallback(
    async (projectId = activeProject?.id, { signal } = {}) => {
      if (!projectId) {
        return;
      }
      try {
        const items = await apiFetch(
          `/api/v1/projects/${projectId}/voices`,
          token,
          { signal },
        );
        // Drop a stale response for a project the user already switched away from
        // (mirrors refreshCharacters' guard).
        if (activeProjectRef?.current?.id && activeProjectRef.current.id !== projectId) {
          return;
        }
        setSavedVoices(Array.isArray(items) ? items : []);
        setError("");
      } catch (err) {
        if (isAbortError(err)) return;
        setError(err.message);
      }
    },
    [token, activeProject, activeProjectRef, setError],
  );

  // Register a saved voice: the backend resolves the reference clip, computes its embedding, runs
  // near-duplicate detection, and persists. Returns the created voice (with a `nearDuplicate` field)
  // so the caller can surface the dedup warning, or null on error.
  const createSavedVoice = useCallback(
    async ({ name, referenceAudioAssetId }) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const created = await apiFetch(
          `/api/v1/projects/${activeProject.id}/voices`,
          token,
          {
            method: "POST",
            body: JSON.stringify({ name, referenceAudioAssetId }),
          },
        );
        setSavedVoices((items) => [
          created,
          ...items.filter((item) => item.id !== created.id),
        ]);
        setError("");
        return created;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, setError],
  );

  const deleteSavedVoice = useCallback(
    async (voiceId) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        await apiFetch(
          `/api/v1/projects/${activeProject.id}/voices/${encodeURIComponent(voiceId)}`,
          token,
          { method: "DELETE" },
        );
        setSavedVoices((items) => items.filter((item) => item.id !== voiceId));
        setError("");
        return { id: voiceId, status: "deleted" };
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, setError],
  );

  return {
    savedVoices,
    setSavedVoices,
    refreshSavedVoices,
    createSavedVoice,
    deleteSavedVoice,
  };
}
