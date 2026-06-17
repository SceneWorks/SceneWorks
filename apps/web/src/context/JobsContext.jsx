import React, { createContext, useCallback, useContext, useEffect, useMemo, useSyncExternalStore } from "react";
import { apiFetch, eventUrl } from "../api.js";
import { AppContext } from "./AppContext.js";
import { DEFAULT_JOB_PROMPT, deriveJobsSnapshot } from "../jobs/jobStore.js";

const JobsStoreContext = createContext(null);
const JobsActionsContext = createContext(null);

const emptySubscribe = () => () => {};
const noop = () => {};

export const selectJobs = (state) => state.jobs;
export const selectFilteredJobs = (state) => state.filteredJobs;
export const selectQueueCounts = (state) => state.queueCounts;
export const selectVisibleWorkers = (state) => state.visibleWorkers;
export const selectWorkersById = (state) => state.workersById;
export const selectGpuOptions = (state) => state.gpuOptions;
export const selectJobPrompt = (state) => state.jobPrompt;
export const selectProjectFilter = (state) => state.projectFilter;
export const selectPersonReadiness = (state) => state.personReadiness;
export const selectImageLocalJobs = (state) => state.imageLocalJobs;
export const selectVideoLocalJobs = (state) => state.videoLocalJobs;
export const selectDocumentLocalJobs = (state) => state.documentLocalJobs;

function legacyJobsSnapshot(value) {
  const workers = value?.workers ?? value?.visibleWorkers ?? [];
  const jobs = value?.jobs ?? value?.filteredJobs ?? [];
  const snapshot = deriveJobsSnapshot({
    jobs,
    workers,
    queueSummary: value?.queueSummary ?? null,
    localGenerationJobIds: value?.localGenerationJobIds ?? {
      image: value?.imageLocalJobs?.map((job) => job.id).filter(Boolean) ?? [],
      video: value?.videoLocalJobs?.map((job) => job.id).filter(Boolean) ?? [],
      document: value?.documentLocalJobs?.map((job) => job.id).filter(Boolean) ?? [],
    },
    activeProjectId: value?.activeProject?.id ?? null,
    projectFilter: value?.projectFilter ?? "all",
    jobPrompt: value?.jobPrompt ?? DEFAULT_JOB_PROMPT,
  });
  return {
    ...snapshot,
    filteredJobs: value?.filteredJobs ?? snapshot.filteredJobs,
    gpuOptions: value?.gpuOptions ?? snapshot.gpuOptions,
    imageLocalJobs: value?.imageLocalJobs ?? snapshot.imageLocalJobs,
    videoLocalJobs: value?.videoLocalJobs ?? snapshot.videoLocalJobs,
    documentLocalJobs: value?.documentLocalJobs ?? snapshot.documentLocalJobs,
    queueCounts: value?.queueCounts ?? snapshot.queueCounts,
    visibleWorkers: value?.visibleWorkers ?? snapshot.visibleWorkers,
    workersById: value?.workersById ?? snapshot.workersById,
  };
}

function legacyJobActions(value) {
  return {
    createPlaceholderJob: value?.createPlaceholderJob ?? noop,
    jobAction: value?.jobAction ?? noop,
    rememberLocalGenerationJob: value?.rememberLocalGenerationJob ?? noop,
    setJobPrompt: value?.setJobPrompt ?? noop,
    setProjectFilter: value?.setProjectFilter ?? noop,
  };
}

export function JobsProvider({ store, actions = {}, children }) {
  const mergedActions = useMemo(
    () => ({
      ...store.actions,
      ...actions,
    }),
    [actions, store],
  );
  return (
    <JobsStoreContext.Provider value={store}>
      <JobsActionsContext.Provider value={mergedActions}>{children}</JobsActionsContext.Provider>
    </JobsStoreContext.Provider>
  );
}

export function useJobsStore() {
  const store = useContext(JobsStoreContext);
  if (!store) {
    throw new Error("useJobsStore must be used within a <JobsProvider>");
  }
  return store;
}

export function useJobsSelector(selector) {
  const store = useContext(JobsStoreContext);
  const legacyContext = useContext(AppContext);
  const legacySnapshot = useMemo(() => legacyJobsSnapshot(legacyContext), [legacyContext]);
  const subscribe = useCallback((listener) => (store ? store.subscribe(listener) : emptySubscribe(listener)), [store]);
  const getSnapshot = useCallback(
    () => selector(store ? store.getSnapshot() : legacySnapshot),
    [legacySnapshot, selector, store],
  );
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export function useJobActions() {
  const contextActions = useContext(JobsActionsContext);
  const legacyContext = useContext(AppContext);
  const fallbackActions = useMemo(() => legacyJobActions(legacyContext), [legacyContext]);
  return contextActions ?? fallbackActions;
}

function parseSseJson(event, label) {
  try {
    return JSON.parse(event.data);
  } catch (err) {
    console.warn(`Ignoring malformed ${label} SSE event`, err);
    return null;
  }
}

function generatedResultAssetCount(job) {
  if (Array.isArray(job.result?.assetIds)) {
    return job.result.assetIds.length;
  }
  if (Array.isArray(job.result?.assets)) {
    return job.result.assets.length;
  }
  return 0;
}

function failedJobNotice(job) {
  const label = String(job.type ?? "job").replaceAll("_", " ");
  const detail = job.error || job.message || "Failed without additional worker detail.";
  return `${label}: ${detail}`;
}

function noticeKindForJob(job) {
  if (job?.type === "lora_import") return "lora-import";
  if (job?.type === "lora_train") return "lora-train";
  return "general";
}

export function JobsEventBridge({
  accessAuthRequired,
  activeProjectRef,
  activeViewRef,
  authenticated,
  dismissNoticeKind,
  enqueueTimelineGenerationApplyRef,
  pushNotice,
  refreshAssetsRef,
  refreshDataRef,
  refreshDataWithLoraOverlayRef,
  refreshPersonTracksRef,
  setError,
  setLatestGenerationSetId,
  token,
}) {
  const store = useJobsStore();

  useEffect(() => {
    if (!authenticated) {
      return undefined;
    }

    let events = null;
    let reconnectTimer = null;
    let reconnectAttempt = 0;
    let closed = false;
    const generatedAssetRefreshes = new Map();

    function handleJobUpdated(event) {
      const job = parseSseJson(event, "job");
      if (!job) {
        return;
      }
      const hasGeneratedAssets = Boolean(job.result?.generationSetId || job.result?.assetIds?.length || job.result?.assets?.length);
      const resultAssetCount = generatedResultAssetCount(job);
      const generationSetId = job.result?.generationSetId ?? "";
      const refreshKey = job.id ?? generationSetId;
      const previousRefresh = generatedAssetRefreshes.get(refreshKey) ?? { assetCount: 0, generationSetId: "" };
      const shouldRefreshGeneratedAssets =
        Boolean(job.projectId) &&
        hasGeneratedAssets &&
        (resultAssetCount > previousRefresh.assetCount ||
          (resultAssetCount === 0 && generationSetId && generationSetId !== previousRefresh.generationSetId));
      store.actions.upsertJob(job);
      if (hasGeneratedAssets) {
        if (job.result?.generationSetId) {
          setLatestGenerationSetId(job.result.generationSetId);
        }
        generatedAssetRefreshes.set(refreshKey, {
          assetCount: Math.max(resultAssetCount, previousRefresh.assetCount),
          generationSetId: generationSetId || previousRefresh.generationSetId,
        });
        if (shouldRefreshGeneratedAssets) {
          refreshAssetsRef.current?.(job.projectId, { mode: "merge" });
        }
      }
      if (job.status === "completed" && hasGeneratedAssets) {
        enqueueTimelineGenerationApplyRef.current?.(job);
      }
      if (job.status === "completed" && job.projectId && job.type === "person_track") {
        refreshPersonTracksRef.current?.(job.projectId);
      }
      if (job.status === "completed" && job.projectId && job.type === "person_detect") {
        refreshAssetsRef.current?.(job.projectId, { mode: "merge" });
      }
      if (job.status === "completed" && job.type === "model_download") {
        refreshDataRef.current?.();
      }
      if (job.status === "completed" && job.type === "lora_download") {
        refreshDataWithLoraOverlayRef.current?.(job.projectId ?? activeProjectRef.current?.id);
      }
      if (job.status === "completed" && job.type === "lora_import") {
        dismissNoticeKind("lora-import");
        refreshDataWithLoraOverlayRef.current?.(job.projectId ?? activeProjectRef.current?.id);
      }
      if (job.status === "completed" && job.type === "lora_train" && job.payload?.dryRun === false) {
        if (job.result?.loraRegistered === false) {
          pushNotice("lora-train", `lora training: ${job.result?.loraRegistrationError ?? "Completed training but could not register the LoRA."}`);
        } else {
          dismissNoticeKind("lora-train");
          refreshDataWithLoraOverlayRef.current?.(job.projectId ?? activeProjectRef.current?.id);
        }
      }
      if (job.status === "failed" && !store.actions.hasVisibleLocalFailure(job, activeViewRef.current)) {
        pushNotice(noticeKindForJob(job), failedJobNotice(job));
      }
    }

    function handleWorkerUpdated(event) {
      const worker = parseSseJson(event, "worker");
      if (!worker) {
        return;
      }
      store.actions.setWorkers((items) => [worker, ...items.filter((item) => item.id !== worker.id)]);
    }

    function handleQueueUpdated(event) {
      const summary = parseSseJson(event, "queue");
      if (!summary) {
        return;
      }
      store.actions.setQueueSummary(summary);
      if (Array.isArray(summary.workers)) {
        store.actions.setWorkers(summary.workers);
      }
    }

    async function connect() {
      let ticket = "";
      try {
        if (accessAuthRequired) {
          const response = await apiFetch("/api/v1/jobs/events/ticket", token, { method: "POST" });
          ticket = response.ticket;
        }
      } catch (err) {
        setError(err.message);
        if (!closed) {
          const delay = Math.min(30000, 1000 * 2 ** reconnectAttempt);
          reconnectAttempt += 1;
          reconnectTimer = window.setTimeout(connect, delay);
        }
        return;
      }

      if (closed) {
        return;
      }

      const source = new EventSource(eventUrl("/api/v1/jobs/events", ticket));
      events = source;
      source.addEventListener("job.updated", handleJobUpdated);
      source.addEventListener("worker.updated", handleWorkerUpdated);
      source.addEventListener("queue.updated", handleQueueUpdated);
      source.onopen = () => {
        reconnectAttempt = 0;
      };
      source.onerror = () => {
        source.close();
        if (closed) {
          return;
        }
        const delay = Math.min(30000, 1000 * 2 ** reconnectAttempt);
        reconnectAttempt += 1;
        reconnectTimer = window.setTimeout(connect, delay);
      };
    }

    connect();

    return () => {
      closed = true;
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }
      events?.close();
    };
  }, [
    accessAuthRequired,
    activeProjectRef,
    activeViewRef,
    authenticated,
    dismissNoticeKind,
    enqueueTimelineGenerationApplyRef,
    pushNotice,
    refreshAssetsRef,
    refreshDataRef,
    refreshDataWithLoraOverlayRef,
    refreshPersonTracksRef,
    setError,
    setLatestGenerationSetId,
    store,
    token,
  ]);

  return null;
}
