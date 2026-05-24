import { useState } from "react";
import { apiFetch, isAbortError } from "../api.js";

// Owns the project's person-track state plus detection/track job creation and manual
// track corrections. Extracted from App.jsx (sc-1651). personTracks is project-scoped
// (loaded by the project-load effect, not the bulk refreshData), so the hook just
// takes the shared concerns it needs; detection/track jobs refresh via refreshData.
export function usePersonTracks({ token, activeProject, setError, requestedGpu, setActiveView, refreshData }) {
  const [personTracks, setPersonTracks] = useState([]);

  async function refreshPersonTracks(projectId = activeProject?.id, { signal } = {}) {
    if (!projectId) {
      return;
    }
    try {
      const items = await apiFetch(`/api/v1/projects/${projectId}/person-tracks`, token, { signal });
      setPersonTracks(items);
      setError("");
    } catch (err) {
      if (isAbortError(err)) return;
      setError(err.message);
    }
  }

  async function createPersonDetectionJob(payload, options = {}) {
    const { navigateToQueue = false } = options;
    if (!activeProject) {
      setError("Create or open a project first.");
      return null;
    }
    try {
      const job = await apiFetch(`/api/v1/projects/${activeProject.id}/person-tracks/detections`, token, {
        method: "POST",
        body: JSON.stringify({ ...payload, requestedGpu }),
      });
      if (navigateToQueue) {
        setActiveView("Queue");
      }
      setError("");
      refreshData();
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  async function createPersonTrackJob(payload, options = {}) {
    const { navigateToQueue = false } = options;
    if (!activeProject) {
      setError("Create or open a project first.");
      return null;
    }
    try {
      const job = await apiFetch(`/api/v1/projects/${activeProject.id}/person-tracks/jobs`, token, {
        method: "POST",
        body: JSON.stringify({ ...payload, requestedGpu }),
      });
      if (navigateToQueue) {
        setActiveView("Queue");
      }
      setError("");
      refreshData();
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  async function saveTrackCorrections(trackId, corrections) {
    if (!activeProject) {
      setError("Create or open a project first.");
      return null;
    }
    try {
      const track = await apiFetch(
        `/api/v1/projects/${activeProject.id}/person-tracks/${trackId}/corrections`,
        token,
        {
          method: "POST",
          body: JSON.stringify({ corrections }),
        },
      );
      setPersonTracks((items) => items.map((item) => (item.id === track.id ? track : item)));
      setError("");
      return track;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }

  return {
    personTracks,
    setPersonTracks,
    refreshPersonTracks,
    createPersonDetectionJob,
    createPersonTrackJob,
    saveTrackCorrections,
  };
}
