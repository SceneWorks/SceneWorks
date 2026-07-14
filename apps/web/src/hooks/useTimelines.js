import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { apiFetch, isAbortError } from "../api.js";
import { ensureItemVersionFields } from "../timeline.js";

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
function timelineGenerationTargetPresent(timeline, job) {
  const payload = job.payload ?? {};
  const action = payload.advanced?.timelineAction;
  const context = payload.advanced?.timelineContext ?? {};
  if (!action || !timeline || context.timelineId !== timeline.id) {
    return false;
  }
  const track = (timeline.tracks ?? []).find((candidate) => candidate.id === context.trackId);
  if (!track) {
    return false;
  }
  if (action === "replace") {
    return (track.items ?? []).some((item) => item.id === context.itemId);
  }
  // extend / bridge only need the target track to exist — they append a new item to it.
  return true;
}

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
  const [activeTimeline, setActiveTimeline] = useState(null);
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

  useEffect(() => {
    activeTimelineRef.current = activeTimeline;
  }, [activeTimeline]);

  // Compare a timeline against the last-persisted snapshot. Unknown baseline (nothing loaded
  // yet) or no working copy → not dirty, so a freshly loaded/empty editor never prompts.
  function timelineHasUnsavedEdits(timeline) {
    if (!timeline || savedTimelineSnapshotRef.current == null) {
      return false;
    }
    return serializeTimeline(timeline) !== savedTimelineSnapshotRef.current;
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
        return;
      }
      try {
        const items = await apiFetch(`/api/v1/projects/${projectId}/timelines`, token, { signal });
        if (activeProjectRef.current?.id && activeProjectRef.current.id !== projectId) {
          return;
        }
        setTimelines(items);
        setTimelinesProjectId(projectId);
        setSelectedTimelineId((current) => (items.some((item) => item.id === current) ? current : items[0]?.id ?? null));
        if (!items.length) {
          setActiveTimeline(null);
          savedTimelineSnapshotRef.current = null;
        }
        setError("");
      } catch (err) {
        if (isAbortError(err)) return;
        setError(err.message);
      }
    },
    [token, activeProject, activeProjectRef, setError],
  );

  async function loadTimeline(projectId, timelineId) {
    try {
      const timeline = await apiFetch(`/api/v1/projects/${projectId}/timelines/${timelineId}`, token);
      if (activeProjectRef.current?.id !== projectId || selectedTimelineIdRef.current !== timelineId) {
        return;
      }
      setActiveTimeline(timeline);
      activeTimelineRef.current = timeline;
      savedTimelineSnapshotRef.current = serializeTimeline(timeline);
      // sc-12018: a fresh load pulls the server copy (which received any conflicting
      // generation), so a prior edit/generation conflict notice is now stale — clear it.
      pushNotice?.(TIMELINE_GENERATION_CONFLICT_NOTICE, "");
      setError("");
    } catch (err) {
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
    [token, activeProject, setError],
  );

  const saveTimeline = useCallback(
    async (timeline) => {
      if (!activeProject || !timeline) {
        return null;
      }
      try {
        const saved = await apiFetch(`/api/v1/projects/${activeProject.id}/timelines/${timeline.id}`, token, {
          method: "PUT",
          body: JSON.stringify({ timeline }),
        });
        setActiveTimeline(saved);
        // sc-11967: reset the dirty baseline to the persisted copy so the timeline reads
        // clean immediately after save (the S7 sibling bug left the baseline stale, so the
        // re-select/SSE guards kept firing on an already-saved timeline).
        activeTimelineRef.current = saved;
        savedTimelineSnapshotRef.current = serializeTimeline(saved);
        // sc-12018: the saved copy is now authoritative, so any prior edit/generation conflict
        // notice ("saving will discard it") has been resolved one way or the other — clear it.
        pushNotice?.(TIMELINE_GENERATION_CONFLICT_NOTICE, "");
        refreshTimelines(activeProject.id);
        setError("");
        return saved;
      } catch (err) {
        setError(err.message);
        return null;
      }
    },
    [token, activeProject, setError, pushNotice, refreshTimelines],
  );

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

  function applyTimelineGenerationResult(timeline, job) {
    const payload = job.payload ?? {};
    const action = payload.advanced?.timelineAction;
    const context = payload.advanced?.timelineContext ?? {};
    const assetId = job.result?.assetIds?.[0];
    if (!action || !assetId || context.timelineId !== timeline.id) {
      return timeline;
    }
    const resultAsset = job.result?.assets?.[0];
    const displayName = resultAsset?.displayName ?? "Generated clip";
    const createdAt = resultAsset?.createdAt ?? new Date().toISOString();
    const tracks = timeline.tracks.map((track) => {
      if (track.id !== context.trackId) {
        return track;
      }
      if (action === "bridge") {
        const bridgeItem = ensureItemVersionFields({
          id: `item_${crypto.randomUUID().replaceAll("-", "")}`,
          trackId: track.id,
          assetId,
          type: "video",
          displayName,
          sourceIn: 0,
          sourceOut: Number(payload.duration) || Math.max(0.1, Number(context.timelineEnd) - Number(context.timelineStart)),
          timelineStart: Number(context.timelineStart),
          timelineEnd: Number(context.timelineEnd),
          speed: 1,
          fit: "fit",
          volume: 1,
          versionAssetIds: [assetId],
          currentVersionAssetId: assetId,
          versionHistory: [{ assetId, createdAt, source: "bridge", jobId: job.id, note: "Generated bridge clip" }],
          transitionIn: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
          transitionOut: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
        });
        return { ...track, items: [...track.items, bridgeItem] };
      }
      if (action === "extend") {
        const start = Number(context.timelineStart);
        const duration = Number(payload.duration) || 4;
        const extensionItem = ensureItemVersionFields({
          id: `item_${crypto.randomUUID().replaceAll("-", "")}`,
          trackId: track.id,
          assetId,
          type: "video",
          displayName,
          sourceIn: 0,
          sourceOut: duration,
          timelineStart: start,
          timelineEnd: start + duration,
          speed: 1,
          fit: "fit",
          volume: 1,
          versionAssetIds: [assetId],
          currentVersionAssetId: assetId,
          versionHistory: [{ assetId, createdAt, source: "extension", jobId: job.id, note: "Generated extension" }],
          transitionIn: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
          transitionOut: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
        });
        return { ...track, items: [...track.items, extensionItem] };
      }
      if (action === "replace") {
        return {
          ...track,
          items: track.items.map((item) => {
            if (item.id !== context.itemId) {
              return item;
            }
            const current = ensureItemVersionFields(item);
            return {
              ...current,
              assetId,
              currentVersionAssetId: assetId,
              type: "video",
              displayName,
              versionAssetIds: Array.from(new Set([...current.versionAssetIds, assetId])),
              versionHistory: [
                ...current.versionHistory,
                { assetId, createdAt, source: "replacement", jobId: job.id, note: "Generated replacement" },
              ],
            };
          }),
        };
      }
      return track;
    });
    return { ...timeline, tracks };
  }

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
      // Always apply the generation onto the server copy and persist it, so the result is
      // durable regardless of what the in-memory working copy looks like (never drop the
      // generation).
      const timeline = await apiFetch(`/api/v1/projects/${projectId}/timelines/${timelineId}`, token);
      const updated = applyTimelineGenerationResult(timeline, job);
      if (updated === timeline) {
        return;
      }
      const saved = await apiFetch(`/api/v1/projects/${projectId}/timelines/${timelineId}`, token, {
        method: "PUT",
        body: JSON.stringify({ timeline: updated }),
      });
      // Preserve the existing gate: only touch the UI copy when this timeline is the one the
      // user is looking at. Re-read the live working copy *after* the awaits — the user may
      // have edited it while the fetch/save was in flight.
      if (selectedTimelineIdRef.current === timelineId) {
        const workingCopy = activeTimelineRef.current;
        if (workingCopy?.id === timelineId && timelineHasUnsavedEdits(workingCopy)) {
          // sc-11967 (S8): the active copy has unsaved structural edits. Overwriting it with
          // the server copy would silently discard them, so merge the generation onto the
          // working copy instead. The baseline is left untouched, so the copy stays dirty and
          // the user still owns the decision to save their edits.
          if (timelineGenerationTargetPresent(workingCopy, job)) {
            const merged = applyTimelineGenerationResult(workingCopy, job);
            setActiveTimeline(merged);
            activeTimelineRef.current = merged;
          } else {
            // sc-12018 (S8 follow-up): the dirty working copy has DELETED the item/track this
            // generation targets, so merging is a content no-op — the generation would be
            // invisible in the live editor. It IS durably persisted to the server copy above,
            // but if the user now SAVES their conflicting deletion the server copy is
            // overwritten and the generation is lost. Surface the conflict instead of silently
            // swallowing it; the working copy is left untouched so the user's deletion — and
            // the ownership of the save decision — is preserved.
            const action = job.payload?.advanced?.timelineAction;
            const clipName = job.result?.assets?.[0]?.displayName;
            const target = action === "replace" ? "a clip you removed" : "a track you removed";
            pushNotice?.(
              TIMELINE_GENERATION_CONFLICT_NOTICE,
              `A generation finished for ${target} from this timeline${clipName ? ` (“${clipName}”)` : ""}. ` +
                "It was saved to the stored timeline, but saving your current edits will discard it. " +
                "Re-open this timeline to keep the generated clip.",
            );
          }
        } else {
          // Clean (or the item the generation targets is gone) → adopt the persisted server
          // copy and reset the baseline so the timeline stays clean (no spurious dirty on a
          // later re-select). This is the pre-sc-11967 behavior.
          setActiveTimeline(saved);
          activeTimelineRef.current = saved;
          savedTimelineSnapshotRef.current = serializeTimeline(saved);
        }
      }
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
    exportTimeline,
    extractTimelineFrame,
    queueTimelineVideoJob,
    enqueueTimelineGenerationApply,
  };
}
