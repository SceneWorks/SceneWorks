import { appConfirm } from "../appConfirm.jsx";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { isCurrentProjectRequest } from "../appStateHelpers.js";
import { refreshFailure, refreshSuccess } from "../refreshResult.js";
import { rebaseTimeline, timelineEdit } from "../timelineEdits.js";

// Canonical serialization of a persisted-timeline snapshot for dirty comparison (sc-11967).
// Value equality (not reference) so a server round-trip that re-emits the same content is
// still "clean". Null timeline → null baseline.
function serializeTimeline(timeline) {
  return timeline ? JSON.stringify(timeline) : null;
}

// sc-12018: the notice `kind` for the edit/generation conflict (a completed generation whose
// target the dirty working copy has removed). A dedicated kind so it neither clobbers nor is
// clobbered by the "general" error notice, and can be cleared once the conflict is resolved.
const TIMELINE_GENERATION_CONFLICT_NOTICE = "timelineGenerationConflict";

// sc-12018: whether the generation's target still exists in `timeline`. A replace rewrites a
// specific item (context.itemId) inside its track; an extend/bridge only appends a new item to
// the target track (context.trackId). applyTimelineGenerationResult ALWAYS returns a fresh
// object when the timelineId matches (it maps `tracks`), so a reference check cannot detect a
// content no-op — this presence check can. Absent target ⇒ applying the generation to this
// copy changes nothing, so it would silently vanish from the live editor.
// Owns the editor's timeline state (list, selection, the loaded timeline) plus every
// timeline mutation, frame extraction, and the SSE-driven "apply generated clip to the
// timeline" pipeline. Extracted from App.jsx (sc-1651) — the largest, most coupled
// slice. App keeps the SSE job.updated handler and calls the returned
// enqueueTimelineGenerationApply; the bulk reset/project-load effects use the returned
// setters/refreshTimelines. createVideoJob (App-owned) is injected for the timeline's
// generate-clip action. The two timeline-specific effects (selectedTimelineId ref sync,
// then the load-on-selection effect) live here, in that order, matching App's prior
// behavior.
export function useTimelines({
  token,
  activeProject,
  activeProjectRef,
  setError,
  pushNotice,
  requestedGpu,
  setActiveView,
  createVideoJob,
}) {
  const [timelines, setTimelines] = useState([]);
  const [timelinesProjectId, setTimelinesProjectId] = useState(null);
  const [selectedTimelineId, setSelectedTimelineId] = useState(null);
  const [activeTimeline, setActiveTimelineState] = useState(null);
  const selectedTimelineIdRef = useRef(null);
  const timelineApplyQueueRef = useRef(Promise.resolve());
  // sc-11967 (S8): the active timeline survives soft navigation in memory, so the user can
  // accumulate unsaved structural edits. Two async paths (the SSE "generation ready" apply
  // and dropdown re-select) used to overwrite `activeTimeline` with the server copy and
  // silently drop those edits. To detect "has unsaved edits" without a bespoke dirty flag
  // scattered across every mutation, we snapshot the last *persisted* timeline (serialized)
  // at each clean baseline point (load / create / save / clean SSE-adopt) and compare the
  // live working copy against it. `activeTimelineRef` mirrors the committed state so the
  // async apply queue and the exposed dirty check read the freshest working copy.
  const savedTimelineSnapshotRef = useRef(null);
  const activeTimelineRef = useRef(null);

  const setActiveTimeline = useCallback((value) => {
    const next = typeof value === "function" ? value(activeTimelineRef.current) : value;
    activeTimelineRef.current = next;
    setActiveTimelineState(next);
  }, []);

  function adoptStoredTimeline(stored) {
    const working = activeTimelineRef.current;
    const base = savedTimelineSnapshotRef.current ? JSON.parse(savedTimelineSnapshotRef.current) : null;
    if (working?.id !== stored.id || !base || base.id !== stored.id) {
      setActiveTimeline(stored);
      savedTimelineSnapshotRef.current = serializeTimeline(stored);
      return;
    }
    if ((stored.revision ?? 0) < (base.revision ?? 0)) return;
    const rebased = rebaseTimeline(base, working, stored);
    if (rebased.conflicts.length) {
      // Keep the local edits and their old base until Save opens the resolver. Never silently
      // accept either side of a same-item race during polling.
      pushNotice?.(TIMELINE_GENERATION_CONFLICT_NOTICE,
        `A generation or stored edit changed clips: ${rebased.conflicts.map((c) => c.label).join(", ")}. Save to choose your edits or the stored versions.`);
      setActiveTimeline(rebaseTimeline(base, working, stored, { force: true }).timeline);
      return;
    }
    setActiveTimeline(rebased.timeline);
    savedTimelineSnapshotRef.current = serializeTimeline(stored);
  }

  const isFilmTimeline = Boolean(activeTimeline?.filmAssembly);

  // Film delivery is server-owned and can finish without an editor SSE subscriber. A bounded
  // read refresh discovers it while preserving pending edits; navigation does not own the run.
  useEffect(() => {
    if (!activeProject?.id || !selectedTimelineId || !isFilmTimeline) return;
    let disposed = false;
    let inFlight = false;
    const timer = window.setInterval(async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const stored = await apiFetch(`/api/v1/projects/${activeProject.id}/timelines/${selectedTimelineId}`, token);
        if (!disposed && activeProjectRef.current?.id === activeProject.id && selectedTimelineIdRef.current === selectedTimelineId) adoptStoredTimeline(stored);
      } catch (err) { if (!disposed) setError(err.message); }
      finally { inFlight = false; }
    }, 1000);
    return () => { disposed = true; window.clearInterval(timer); };
    // Ref-backed adoption reads current pending edits, never the timer's initial snapshot.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeProject?.id, selectedTimelineId, isFilmTimeline, token]);

  // Compare a timeline against the last-persisted snapshot. Unknown baseline (nothing loaded
  // yet) or no working copy → not dirty, so a freshly loaded/empty editor never prompts.
  function timelineHasUnsavedEdits(timeline) {
    if (!timeline || savedTimelineSnapshotRef.current == null) {
      return false;
    }
    return timelineEdit(JSON.parse(savedTimelineSnapshotRef.current), timeline).length > 0;
  }

  // Stable identity (reads refs only) so it can ride the memoized App context without
  // re-rendering consumers. The re-select guard (EditorScreen) calls this imperatively.
  const isActiveTimelineDirty = useCallback(() => timelineHasUnsavedEdits(activeTimelineRef.current), []);

  useEffect(() => {
    selectedTimelineIdRef.current = selectedTimelineId;
  }, [selectedTimelineId]);

  useEffect(() => {
    if (!activeProject || !selectedTimelineId || timelinesProjectId !== activeProject.id) {
      return;
    }
    loadTimeline(activeProject.id, selectedTimelineId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeProject?.id, selectedTimelineId, timelinesProjectId]);

  // sc-4194: the context-exposed actions (and refreshTimelines, on which they depend)
  // are wrapped in useCallback so their identity is stable across App's SSE-driven
  // re-renders, letting appContextValue memoize. The internal helpers below
  // (loadTimeline, applyTimelineGenerationResult, enqueueTimelineGenerationApply) are
  // not part of the context value, so they stay plain function declarations — keeping
  // loadTimeline hoisted for the load-on-selection effect above.
  const refreshTimelines = useCallback(
    async (projectId = activeProject?.id, { signal } = {}) => {
      if (!projectId) {
        return refreshFailure("missing-project");
      }
      if (!isCurrentProjectRequest(activeProject?.id ?? null, projectId)) {
        return refreshFailure("stale");
      }
      try {
        const items = await apiFetch(`/api/v1/projects/${projectId}/timelines`, token, { signal });
        if (!isCurrentProjectRequest(activeProjectRef.current?.id ?? null, projectId)) {
          return refreshFailure("stale");
        }
        setTimelines(items);
        setTimelinesProjectId(projectId);
        setSelectedTimelineId((current) => (items.some((item) => item.id === current) ? current : items[0]?.id ?? null));
        if (!items.length) {
          setActiveTimeline(null);
          savedTimelineSnapshotRef.current = null;
        }
        setError("");
        return refreshSuccess(items);
      } catch (err) {
        if (!isCurrentProjectRequest(activeProjectRef.current?.id ?? null, projectId)) {
          return refreshFailure("stale", err);
        }
        if (isAbortError(err)) return refreshFailure("aborted", err);
        setError(err.message);
        return refreshFailure("error", err);
      }
    },
    [token, activeProject, activeProjectRef, setError, setActiveTimeline],
  );

  async function loadTimeline(projectId, timelineId) {
    if (
      !isCurrentProjectRequest(activeProjectRef.current?.id ?? null, projectId) ||
      selectedTimelineIdRef.current !== timelineId
    ) {
      return;
    }
    try {
      const timeline = await apiFetch(`/api/v1/projects/${projectId}/timelines/${timelineId}`, token);
      if (activeProjectRef.current?.id !== projectId || selectedTimelineIdRef.current !== timelineId) {
        return;
      }
      adoptStoredTimeline(timeline);
      // sc-12018: a fresh load pulls the server copy (which received any conflicting
      // generation), so a prior edit/generation conflict notice is now stale — clear it.
      pushNotice?.(TIMELINE_GENERATION_CONFLICT_NOTICE, "");
      setError("");
    } catch (err) {
      if (
        !isCurrentProjectRequest(activeProjectRef.current?.id ?? null, projectId) ||
        selectedTimelineIdRef.current !== timelineId
      ) {
        return;
      }
      setError(err.message);
    }
  }

  const createTimeline = useCallback(
    async (payload) => {
      if (!activeProject) {
        setError("Create or open a project first.");
        return null;
      }
      try {
        const created = await apiFetch(`/api/v1/projects/${activeProject.id}/timelines`, token, {
          method: "POST",
          body: JSON.stringify(payload),
        });
        setTimelines((items) => [created, ...items.filter((item) => item.id !== created.id)]);
        setTimelinesProjectId(activeProject.id);
        setSelectedTimelineId(created.id);
        setActiveTimeline(created);
        activeTimelineRef.current = created;
        savedTimelineSnapshotRef.current = serializeTimeline(created);
        setError("");
        return created;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, setError, setActiveTimeline],
  );

  const saveTimeline = useCallback(
    async (timeline) => {
      if (!activeProject || !timeline) {
        return null;
      }
      const workingAtSave = activeTimelineRef.current;
      try {
        const projectId = activeProject.id;
        const path = `/api/v1/projects/${projectId}/timelines/${timeline.id}`;
        const initialBase = savedTimelineSnapshotRef.current ? JSON.parse(savedTimelineSnapshotRef.current) : timeline;
        let base = initialBase?.id === timeline.id ? initialBase : timeline;
        let candidate = timeline;
        let saved;
        for (let attempt = 0; attempt < 4; attempt += 1) {
          try {
            saved = await apiFetch(path, token, { method: "PUT", body: JSON.stringify({ timeline: candidate, expectedRevision: base.revision ?? 0 }) });
            break;
          } catch (error) {
            if (error.code !== "timeline_revision_conflict") throw error;
            const stored = await apiFetch(path, token);
            let merged = rebaseTimeline(base, candidate, stored);
            if (merged.conflicts.length) {
              const names = merged.conflicts.map((c) => c.label).join(", ");
              const keepLocal = await appConfirm({ title: "Resolve timeline conflict", message: `These clips changed in the stored timeline: ${names}. Unrelated edits are preserved.`, confirmLabel: "Keep my versions", cancelLabel: "Keep stored versions" });
              if (keepLocal) merged = rebaseTimeline(base, candidate, stored, { force: true });
            }
            candidate = merged.timeline;
            base = stored;
          }
        }
        if (!saved) throw new Error("The timeline keeps changing. Your edits are still here; save again to resolve the latest version.");
        if (activeProjectRef.current?.id !== projectId || selectedTimelineIdRef.current !== timeline.id) return saved;
        // A user can continue editing while this request is running. Rebase those later edits
        // over the result instead of treating the response as permission to erase them.
        const live = activeTimelineRef.current;
        const afterSave = live?.id === timeline.id ? rebaseTimeline(workingAtSave ?? timeline, live, saved) : { timeline: saved, conflicts: [] };
        setActiveTimeline(afterSave.conflicts.length ? rebaseTimeline(workingAtSave ?? timeline, live, saved, { force: true }).timeline : afterSave.timeline);
        // sc-11967: reset the dirty baseline to the persisted copy so the timeline reads
        // clean immediately after save (the S7 sibling bug left the baseline stale, so the
        // re-select/SSE guards kept firing on an already-saved timeline).
        savedTimelineSnapshotRef.current = serializeTimeline(afterSave.conflicts.length ? initialBase : saved);
        // sc-12018: the saved copy is now authoritative, so any prior edit/generation conflict
        // notice ("saving will discard it") has been resolved one way or the other — clear it.
        pushNotice?.(TIMELINE_GENERATION_CONFLICT_NOTICE, afterSave.conflicts.length ? "A clip changed while saving. Your newer edits remain pending; save again to resolve them." : "");
        refreshTimelines(activeProject.id);
        setError("");
        return saved;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, activeProjectRef, setActiveTimeline, setError, pushNotice, refreshTimelines],
  );

  const resolveTimelineTrim = useCallback(async (runId, shotId, resolution) => {
    const working = activeTimelineRef.current;
    if (!working || !activeProject) return;
    const saved = await saveTimeline(working);
    if (!saved) return;
    if (!runId) {
      try {
        const { resolveGenerationTrim } = await import("../timelineGeneration.js");
        const next = resolveGenerationTrim(saved, shotId, resolution);
        if (next) await saveTimeline(next);
      } catch (error) { setError(error.message); }
      return;
    }
    try {
      const stored = await apiFetch(`/api/v1/projects/${activeProject.id}/timelines/${saved.id}/film-deliveries`, token, {
        method: "POST", body: JSON.stringify({ runId, resolveShotId: shotId, resolution, expectedRevision: saved.revision ?? 0 }),
      });
      if (activeProjectRef.current?.id === activeProject.id && selectedTimelineIdRef.current === saved.id) adoptStoredTimeline(stored);
    } catch (error) { setError(error.message); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeProject, token, saveTimeline, activeProjectRef, setError]);

  const exportTimeline = useCallback(
    async (timeline, options) => {
      if (!activeProject || !timeline) {
        return;
      }
      const saved = await saveTimeline(timeline);
      if (!saved) {
        return;
      }
      try {
        await apiFetch(`/api/v1/projects/${activeProject.id}/timelines/${saved.id}/exports`, token, {
          method: "POST",
          body: JSON.stringify({ ...options, requestedGpu }),
        });
        setActiveView("Queue");
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token, activeProject, setError, requestedGpu, setActiveView, saveTimeline],
  );

  const extractTimelineFrame = useCallback(
    async ({ timeline, item, playheadSeconds, intendedUse }) => {
      if (!activeProject || !timeline || !item) {
        return null;
      }
      try {
        const job = await apiFetch(`/api/v1/projects/${activeProject.id}/timelines/${timeline.id}/items/${item.id}/frames`, token, {
          method: "POST",
          body: JSON.stringify({ playheadSeconds, intendedUse, requestedGpu }),
        });
        setError("");
        return job;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, setError, requestedGpu],
  );

  const queueTimelineVideoJob = useCallback(
    async (payload) => createVideoJob(payload, { navigateToQueue: false }),
    [createVideoJob],
  );

  // sc-11231 (F-037): useJobEvents' SSE effect captures this at subscribe time (its deps
  // are only [access.authRequired, ready, token]), so it MUST be identity-stable — a plain
  // per-render function declaration left the stream calling a stale closure (the same
  // stale-closure class as the prior F-009 fix). The queue push itself only touches refs,
  // but applyCompletedTimelineGeneration is recreated every render and closes over the live
  // `token`, so we publish it into a ref each commit (the `stableRefreshData` ref-delegation
  // pattern) and expose a stable callback that always runs the freshest body. useLayoutEffect
  // matches App's ref-publish ordering so the ref holds the newest committed closure.
  const applyCompletedTimelineGenerationRef = useRef(null);
  useLayoutEffect(() => {
    applyCompletedTimelineGenerationRef.current = applyCompletedTimelineGeneration;
  });
  const enqueueTimelineGenerationApply = useCallback(
    (job) => {
      timelineApplyQueueRef.current = timelineApplyQueueRef.current
        .then(() => applyCompletedTimelineGenerationRef.current?.(job))
        .catch((err) => setError(err.message));
    },
    [setError],
  );

  async function applyCompletedTimelineGeneration(job) {
    const timelineId = job.payload?.advanced?.timelineContext?.timelineId;
    const projectId = job.projectId;
    if (!projectId || !timelineId || !job.result?.assetIds?.length) {
      return;
    }
    try {
      const { applyTimelineGenerationResult } = await import("../timelineGeneration.js");
      const path = `/api/v1/projects/${projectId}/timelines/${timelineId}`;
      let saved;
      for (let attempt = 0; attempt < 4; attempt += 1) {
        const timeline = await apiFetch(path, token);
        const updated = applyTimelineGenerationResult(timeline, job);
        if (serializeTimeline(updated) === serializeTimeline(timeline)) { saved = timeline; break; }
        try {
          saved = await apiFetch(path, token, { method: "PUT", body: JSON.stringify({ timeline: updated, expectedRevision: timeline.revision ?? 0 }) });
          break;
        } catch (error) { if (error.code !== "timeline_revision_conflict") throw error; }
      }
      if (!saved) throw new Error("The generated clip could not be applied because the timeline kept changing. Reopen the timeline to retry.");
      if (activeProjectRef.current?.id === projectId && selectedTimelineIdRef.current === timelineId) adoptStoredTimeline(saved);
      refreshTimelines(projectId);
    } catch (err) {
      setError(err.message);
    }
  }

  return {
    timelines,
    setTimelines,
    timelinesProjectId,
    setTimelinesProjectId,
    selectedTimelineId,
    setSelectedTimelineId,
    activeTimeline,
    setActiveTimeline,
    isActiveTimelineDirty,
    refreshTimelines,
    createTimeline,
    saveTimeline,
    resolveTimelineTrim,
    exportTimeline,
    extractTimelineFrame,
    queueTimelineVideoJob,
    enqueueTimelineGenerationApply,
  };
}
