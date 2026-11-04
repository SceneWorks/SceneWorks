import { useAppDomainValue } from "../context/useAppDomainValue.js";
import { useProjectScopedList } from "./useProjectScopedList.js";
import { personTracksApi } from "../api/personTracksResources.js";
import { useCallback, useState } from "react";

// Owns the project's person-track state plus detection/track job creation and manual
// track corrections. Extracted from App.jsx (sc-1651). personTracks is project-scoped
// (loaded by the project-load effect, not the bulk refreshData), so the hook just
// takes the shared concerns it needs; detection/track jobs surface live via the SSE
// job stream (the create endpoints publish job.updated), so no post-create refetch.
export function usePersonTracks({ token, activeProject, activeProjectRef, setError, requestedGpu, setActiveView }) {
  const [personTracks, setPersonTracks] = useState([]);

  // sc-4194: every action is wrapped in useCallback so its identity is stable
  // across the SSE-driven re-renders of App, letting appContextValue memoize.
  const refreshPersonTracks = useProjectScopedList({
    token, activeProject, activeProjectRef,
    load: personTracksApi.list,
    setItems: setPersonTracks,
    setError: setError,
  });

  const createPersonDetectionJob = useCallback(
    async (payload, options = {}) => {
      const { navigateToQueue = false } = options;
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const job = await personTracksApi.detect(activeProject.id, token, {
          method: "POST",
          body: JSON.stringify({ ...payload, requestedGpu }),
        });
        if (navigateToQueue) {
          setActiveView("Queue");
        }
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, setError, requestedGpu, setActiveView],
  );

  const createPersonTrackJob = useCallback(
    async (payload, options = {}) => {
      const { navigateToQueue = false } = options;
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const job = await personTracksApi.jobs(activeProject.id, token, {
          method: "POST",
          body: JSON.stringify({ ...payload, requestedGpu }),
        });
        if (navigateToQueue) {
          setActiveView("Queue");
        }
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, setError, requestedGpu, setActiveView],
  );

  const saveTrackCorrections = useCallback(
    async (trackId, corrections) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const track = await personTracksApi.corrections(activeProject.id, trackId, token, {
            method: "POST",
            body: JSON.stringify({ corrections }),
          });
        setPersonTracks((items) => items.map((item) => (item.id === track.id ? track : item)));
        setError("");
        return track;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, setError],
  );

  const context = useAppDomainValue({
    personTracks,
    createPersonDetectionJob,
    createPersonTrackJob,
    saveTrackCorrections
  });
  return { ...context, setPersonTracks, refreshPersonTracks, context };
}
