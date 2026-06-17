import { terminalStatuses } from "../constants.js";
import { sortWorkers } from "../sorters.js";
import { buildWorkersById } from "../workers.js";

export const DEFAULT_JOB_PROMPT = "Placeholder generation";
export const localJobStackLimit = 25;

const emptyLocalGenerationJobIds = Object.freeze({ image: [], video: [], document: [] });

export function isActiveWorker(worker) {
  return worker.status !== "offline";
}

function hasCapability(worker, capability) {
  return Array.isArray(worker.capabilities) && worker.capabilities.includes(capability);
}

export function isPlaceholderOnlyGpuWorker(worker) {
  if (!hasCapability(worker, "gpu")) {
    return false;
  }
  const capabilities = Array.isArray(worker.capabilities) ? worker.capabilities : [];
  return capabilities.every((capability) => ["placeholder", "gpu", "nvidia"].includes(capability));
}

function isSelectableGpuWorker(worker) {
  return worker.gpuId && worker.gpuId !== "cpu" && hasCapability(worker, "gpu") && !isPlaceholderOnlyGpuWorker(worker);
}

function isImageGenerationJob(job) {
  return ["image_generate", "image_edit"].includes(job.type);
}

function isVideoGenerationJob(job) {
  return ["video_generate", "video_extend", "video_bridge"].includes(job.type);
}

function isInterleaveJob(job) {
  return job.type === "image_interleave";
}

function normalizeLocalGenerationJobIds(value) {
  return {
    image: Array.isArray(value?.image) ? value.image : [],
    video: Array.isArray(value?.video) ? value.video : [],
    document: Array.isArray(value?.document) ? value.document : [],
  };
}

function jobFreshnessMs(job) {
  const timestamp = job?.updatedAt ?? job?.completedAt ?? job?.canceledAt ?? job?.startedAt ?? job?.createdAt;
  const parsed = Date.parse(timestamp ?? "");
  return Number.isFinite(parsed) ? parsed : 0;
}

function sortJobNewest(a, b) {
  return String(b.createdAt ?? "").localeCompare(String(a.createdAt ?? ""));
}

function sortJobOldest(a, b) {
  return String(a.createdAt ?? "").localeCompare(String(b.createdAt ?? ""));
}

export function mergeFreshJobs(currentJobs, serverJobs) {
  const merged = new Map();
  for (const job of serverJobs ?? []) {
    merged.set(job.id, job);
  }
  for (const current of currentJobs ?? []) {
    const server = merged.get(current.id);
    if (!server || jobFreshnessMs(current) > jobFreshnessMs(server)) {
      merged.set(current.id, current);
    }
  }
  return [...merged.values()].sort(sortJobNewest);
}

function buildLocalJobStack(rememberedIds, jobs, activeProjectId, isGenerationJob) {
  const remembered = rememberedIds.map((id) => jobs.find((job) => job.id === id)).filter(Boolean);
  const projectJobs = jobs.filter(
    (job) =>
      activeProjectId &&
      job.projectId === activeProjectId &&
      isGenerationJob(job) &&
      !terminalStatuses.has(job.status),
  );
  const byId = new Map();
  [...remembered, ...projectJobs].forEach((job) => {
    if (job?.id && !byId.has(job.id)) {
      byId.set(job.id, job);
    }
  });
  return Array.from(byId.values()).sort(sortJobOldest).slice(-localJobStackLimit);
}

function queueCountsFor(jobs, queueSummary) {
  if (queueSummary?.counts) {
    return {
      ...queueSummary.counts,
      active: queueSummary.activeJobs?.length ?? jobs.filter((job) => !terminalStatuses.has(job.status)).length,
    };
  }
  return jobs.reduce(
    (counts, job) => {
      counts[job.status] = (counts[job.status] ?? 0) + 1;
      if (!terminalStatuses.has(job.status)) {
        counts.active += 1;
      }
      return counts;
    },
    { active: 0 },
  );
}

function personReadinessFor(workers) {
  const live = workers.filter((worker) => worker.status !== "offline");
  const ready = (capability) => live.some((worker) => (worker.capabilities ?? []).includes(capability));
  return {
    detect: { capability: "person_detect", ready: ready("person_detect") },
    track: { capability: "person_track", ready: ready("person_track") },
    segment: { capability: "person_segment", ready: ready("person_segment") },
    replace: { capability: "person_replace", ready: ready("person_replace") },
    detectPreview: { capability: "person_detect_preview", ready: ready("person_detect_preview") },
    trackPreview: { capability: "person_track_preview", ready: ready("person_track_preview") },
  };
}

export function deriveJobsSnapshot(baseState) {
  const jobs = Array.isArray(baseState.jobs) ? baseState.jobs : [];
  const workers = Array.isArray(baseState.workers) ? baseState.workers : [];
  const localGenerationJobIds = normalizeLocalGenerationJobIds(baseState.localGenerationJobIds);
  const visibleWorkers = workers.filter((worker) => isActiveWorker(worker) && !isPlaceholderOnlyGpuWorker(worker));
  const gpuIds = visibleWorkers.filter(isSelectableGpuWorker).map((worker) => worker.gpuId);
  const projectFilter = baseState.projectFilter ?? "all";
  return {
    ...baseState,
    jobs,
    workers,
    localGenerationJobIds,
    projectFilter,
    jobPrompt: baseState.jobPrompt ?? DEFAULT_JOB_PROMPT,
    activeProjectId: baseState.activeProjectId ?? null,
    queueSummary: baseState.queueSummary ?? null,
    filteredJobs: projectFilter === "all" ? jobs : jobs.filter((job) => job.projectId === projectFilter),
    queueCounts: queueCountsFor(jobs, baseState.queueSummary),
    visibleWorkers,
    workersById: buildWorkersById(workers),
    gpuOptions: ["auto", ...Array.from(new Set(gpuIds))],
    personReadiness: personReadinessFor(workers),
    imageLocalJobs: buildLocalJobStack(localGenerationJobIds.image, jobs, baseState.activeProjectId, isImageGenerationJob),
    videoLocalJobs: buildLocalJobStack(localGenerationJobIds.video, jobs, baseState.activeProjectId, isVideoGenerationJob),
    documentLocalJobs: buildLocalJobStack(localGenerationJobIds.document, jobs, baseState.activeProjectId, isInterleaveJob),
  };
}

export function createJobsStore(initialState = {}) {
  let state = deriveJobsSnapshot({
    jobs: [],
    workers: [],
    queueSummary: null,
    localGenerationJobIds: emptyLocalGenerationJobIds,
    activeProjectId: null,
    projectFilter: "all",
    jobPrompt: DEFAULT_JOB_PROMPT,
    ...initialState,
  });
  const listeners = new Set();

  function emit() {
    for (const listener of listeners) {
      listener();
    }
  }

  function setBaseState(updater) {
    const nextBase = typeof updater === "function" ? updater(state) : { ...state, ...updater };
    state = deriveJobsSnapshot(nextBase);
    emit();
  }

  function setJobs(updater) {
    setBaseState((current) => ({
      ...current,
      jobs: typeof updater === "function" ? updater(current.jobs) : (updater ?? []),
    }));
  }

  function upsertJob(job) {
    if (!job?.id) {
      return;
    }
    setJobs((items) => [job, ...items.filter((item) => item.id !== job.id)].sort(sortJobNewest));
  }

  function setWorkers(updater) {
    setBaseState((current) => {
      const nextWorkers = typeof updater === "function" ? updater(current.workers) : (updater ?? []);
      return {
        ...current,
        workers: [...nextWorkers].sort(sortWorkers),
      };
    });
  }

  function rememberLocalGenerationJob(kind, job) {
    if (!job?.id || !["image", "video", "document"].includes(kind)) {
      return;
    }
    setBaseState((current) => ({
      ...current,
      localGenerationJobIds: {
        ...current.localGenerationJobIds,
        [kind]: [job.id, ...current.localGenerationJobIds[kind].filter((id) => id !== job.id)].slice(0, localJobStackLimit),
      },
    }));
  }

  function hasVisibleLocalFailure(job, activeView) {
    const localIds = state.localGenerationJobIds;
    if (activeView === "Image" && localIds.image.includes(job.id)) {
      return true;
    }
    if (activeView === "Video" && localIds.video.includes(job.id)) {
      return true;
    }
    if (activeView === "Document" && localIds.document.includes(job.id)) {
      return true;
    }
    return activeView === "Models" && job.type === "model_download";
  }

  const actions = {
    setActiveProjectId: (activeProjectId) => setBaseState((current) => ({ ...current, activeProjectId: activeProjectId ?? null })),
    setJobPrompt: (jobPrompt) => setBaseState((current) => ({ ...current, jobPrompt })),
    setJobs,
    setProjectFilter: (projectFilter) => setBaseState((current) => ({ ...current, projectFilter })),
    setQueueSummary: (queueSummary) => setBaseState((current) => ({ ...current, queueSummary })),
    setWorkers,
    mergeServerJobs: (serverJobs) => setJobs((current) => mergeFreshJobs(current, serverJobs)),
    rememberLocalGenerationJob,
    upsertJob,
    hasVisibleLocalFailure,
  };

  return {
    actions,
    getSnapshot: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
