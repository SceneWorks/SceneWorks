import { useCallback, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { isCurrentProjectRequest } from "../appStateHelpers.js";
import { refreshFailure, refreshSuccess } from "../refreshResult.js";

// Owns the project's Voice Clone "saved voices" roster + its create/delete mutations (sc-13517).
// A saved voice is a named pointer to a library audio reference clip (plus its Chatterbox-VE speaker
// embedding, computed + dedup-checked on the backend at register time). Extracted as a thin data
// layer like useCharacters/usePresets; shared concerns (token, active project, error) are passed in.
//
// Every returned action is useCallback-wrapped with stable deps so appStaticValue can stay memoized
// across App's SSE-driven re-renders (mirrors the sc-4194 note in useCharacters/useModelsAndLoras).
export function useSavedVoices({ token, activeProject, activeProjectRef, setError }) {
  const [savedVoices, setSavedVoices] = useState([]);

  const isCurrentVoiceRequest = useCallback(
    (projectId) =>
      isCurrentProjectRequest(activeProjectRef?.current?.id ?? null, projectId),
    [activeProjectRef],
  );

  const refreshSavedVoices = useCallback(
    async (projectId = activeProject?.id, { signal } = {}) => {
      if (!projectId) {
        return refreshFailure("missing-project");
      }
      if (!isCurrentProjectRequest(activeProject?.id ?? null, projectId)) {
        return refreshFailure("stale");
      }
      try {
        const items = await apiFetch(
          `/api/v1/projects/${projectId}/voices`,
          token,
          { signal },
        );
        // Drop a stale response for a project the user already switched away from
        // (mirrors refreshCharacters' guard).
        if (!isCurrentVoiceRequest(projectId)) {
          return refreshFailure("stale");
        }
        setSavedVoices(Array.isArray(items) ? items : []);
        setError("");
        return refreshSuccess(items);
      } catch (err) {
        if (!isCurrentVoiceRequest(projectId)) return refreshFailure("stale", err);
        if (isAbortError(err)) return refreshFailure("aborted", err);
        setError(err.message);
        return refreshFailure("error", err);
      }
    },
    [token, activeProject, isCurrentVoiceRequest, setError],
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
      const projectId = activeProject.id;
      try {
        const created = await apiFetch(
          `/api/v1/projects/${projectId}/voices`,
          token,
          {
            method: "POST",
            body: JSON.stringify({ name, referenceAudioAssetId }),
          },
        );
        if (!isCurrentVoiceRequest(projectId)) {
          return null;
        }
        setSavedVoices((items) => [
          created,
          ...items.filter((item) => item.id !== created.id),
        ]);
        setError("");
        return created;
      } catch (err) {
        if (!isCurrentVoiceRequest(projectId)) return null;
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, isCurrentVoiceRequest, setError],
  );

  const deleteSavedVoice = useCallback(
    async (voiceId) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      const projectId = activeProject.id;
      try {
        await apiFetch(
          `/api/v1/projects/${projectId}/voices/${encodeURIComponent(voiceId)}`,
          token,
          { method: "DELETE" },
        );
        if (!isCurrentVoiceRequest(projectId)) {
          return null;
        }
        setSavedVoices((items) => items.filter((item) => item.id !== voiceId));
        setError("");
        return { id: voiceId, status: "deleted" };
      } catch (err) {
        if (!isCurrentVoiceRequest(projectId)) return null;
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, isCurrentVoiceRequest, setError],
  );

  return {
    savedVoices,
    setSavedVoices,
    refreshSavedVoices,
    createSavedVoice,
    deleteSavedVoice,
  };
}
