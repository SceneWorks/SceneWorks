import React, { useEffect, useMemo, useState } from "react";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { terminalStatuses } from "../constants.js";
import { GPU_REQUIRED_JOB_TYPES, NON_GPU_JOB_TYPES, pendingStatuses } from "../jobTypes.js";
import { useAppContext } from "../context/AppContext.js";
import { resolveJobResultAssets } from "../jobResultAssets.js";
import { WorkPanel } from "../components/WorkPanel.jsx";

function formatJobType(type) {
  return String(type ?? "job").replaceAll("_", " ");
}

function queueRank(job) {
  const rank = Number(job?.queueRank);
  return Number.isSafeInteger(rank) && rank > 0 ? rank : 0;
}

// Match the order workers observe: already-running work first, then pending work by durable
// priority/FIFO, then recent history. A worker can still skip an incompatible job, but it cannot
// let GPU affinity cross a priority tier.
function queueDisplayOrder(left, right) {
  const group = (job) => {
    if (pendingStatuses.has(job.status)) return 1;
    if (terminalStatuses.has(job.status)) return 2;
    return 0;
  };
  const groupDelta = group(left) - group(right);
  if (groupDelta !== 0) return groupDelta;
  if (pendingStatuses.has(left.status)) {
    const rankDelta = queueRank(right) - queueRank(left);
    if (rankDelta !== 0) return rankDelta;
    return String(left.createdAt ?? "").localeCompare(String(right.createdAt ?? ""));
  }
  return String(right.createdAt ?? "").localeCompare(String(left.createdAt ?? ""));
}

function workerSupports(worker, type) {
  return Array.isArray(worker.capabilities) && worker.capabilities.includes(type);
}

function workerCanClaim(job, worker) {
  if (!workerSupports(worker, job.type)) {
    return false;
  }
  if (NON_GPU_JOB_TYPES.has(job.type)) {
    return true;
  }
  if (GPU_REQUIRED_JOB_TYPES.has(job.type) && worker.gpuId === "cpu") {
    return false;
  }
  return job.requestedGpu === "auto" || job.requestedGpu === worker.gpuId;
}

function modelKeys(job) {
  const keys = new Set();
  if (job.payload?.model) {
    keys.add(job.payload.model);
  }
  if (job.payload?.repo) {
    keys.add(job.payload.repo);
  }
  if (job.payload?.advanced?.modelRepo) {
    keys.add(job.payload.advanced.modelRepo);
  }
  if (job.payload?.advanced?.repo) {
    keys.add(job.payload.advanced.repo);
  }
  return keys;
}

function activeModelDownloadFor(job, jobs) {
  const keys = modelKeys(job);
  if (!keys.size) {
    return null;
  }
  return jobs.find(
    (candidate) =>
      candidate.type === "model_download" &&
      !terminalStatuses.has(candidate.status) &&
      (keys.has(candidate.payload?.modelId) || keys.has(candidate.payload?.repo)),
  );
}

function dependencyJobId(job) {
  return job.payload?.dependsOnJobId ?? job.payload?.dependencyJobId ?? job.dependsOnJobId ?? job.sourceJobId ?? null;
}

function activeDependencyFor(job, jobs) {
  const id = dependencyJobId(job);
  if (!id) {
    return null;
  }
  const dependency = jobs.find((candidate) => candidate.id === id);
  return dependency && !terminalStatuses.has(dependency.status) ? dependency : null;
}

function jobWaitingMessage(job, workers, jobs) {
  if (job.status !== "queued") {
    return job.error ?? job.message;
  }
  const dependency = activeDependencyFor(job, jobs);
  if (dependency) {
    return `Waiting for dependency ${dependency.id} to finish.`;
  }
  const download = activeModelDownloadFor(job, jobs);
  if (download) {
    return `Waiting for model download ${download.payload?.modelName ?? download.payload?.modelId ?? download.id} to finish.`;
  }
  const candidates = workers.filter((worker) => workerCanClaim(job, worker));
  if (!candidates.length) {
    // sc-16260: an unusable GPU is the most likely reason a candle host has NO eligible worker —
    // the worker withheld its capabilities on purpose, so "no active worker supports image
    // generate" is technically true and completely unhelpful. When an unhealthy worker is
    // registered, lead with the host-side remedy it reported. Checked before the requested-GPU
    // branch: a job pinned to the very GPU that failed its probe deserves the reason, not a
    // restatement of the pin.
    //
    // Scoped twice, because a wrong remedy is worse than a vague one:
    //   * NOT for utility work. A queued `model_download` does not need the GPU, so blaming the
    //     driver for it would send an operator to fix something unrelated to why it is stuck.
    //   * When the job pins a GPU, only THAT GPU's worker may explain it — on a multi-GPU host,
    //     gpu 0 being unhealthy says nothing about a job pinned to gpu 1.
    const pinned = job.requestedGpu && job.requestedGpu !== "auto" ? job.requestedGpu : null;
    const unhealthy = NON_GPU_JOB_TYPES.has(job.type)
      ? null
      : workers
          .filter((worker) => pinned === null || worker.gpuId === pinned)
          .map(workerHealthReason)
          .find(Boolean);
    if (unhealthy) {
      return `Blocked: this machine's GPU is unavailable, so ${formatJobType(job.type)} cannot run.\n\n${unhealthy}`;
    }
    if (job.requestedGpu && job.requestedGpu !== "auto") {
      return `Blocked: no active worker can run ${formatJobType(job.type)} on GPU ${job.requestedGpu}.`;
    }
    return `Blocked: no active worker supports ${formatJobType(job.type)}.`;
  }
  if (candidates.every((worker) => worker.status === "busy")) {
    const target = job.requestedGpu && job.requestedGpu !== "auto" ? `GPU ${job.requestedGpu}` : "an eligible worker";
    return `Waiting: ${target} is busy.`;
  }
  if (job.requestedGpu && job.requestedGpu !== "auto") {
    return `Waiting for GPU ${job.requestedGpu} to claim the job.`;
  }
  if (NON_GPU_JOB_TYPES.has(job.type)) {
    return "Waiting for a utility worker.";
  }
  return GPU_REQUIRED_JOB_TYPES.has(job.type)
    ? "Waiting for an available GPU worker."
    : "Waiting for an available worker with the required capability.";
}

function workerStatusLine(worker) {
  if (worker.status === "busy") {
    return `Busy${worker.currentJobId ? ` with ${worker.currentJobId}` : ""}`;
  }
  // sc-16260: a worker whose GPU failed its startup probe has withdrawn every capability it
  // serves, so it will never claim anything. It still heartbeats, so without this it renders the
  // raw enum next to a card that otherwise looks perfectly healthy.
  if (worker.status === "unhealthy") {
    return "GPU unavailable";
  }
  return worker.status === "idle" ? "Ready" : worker.status;
}

/// sc-16260: the host-side remedy an `unhealthy` worker reports, or `null`. The API only sets
/// `statusReason` alongside the unhealthy status, but this checks both so a stale reason left on a
/// recovered worker can never surface as a live warning.
function workerHealthReason(worker) {
  if (worker?.status !== "unhealthy") {
    return null;
  }
  const reason = typeof worker.statusReason === "string" ? worker.statusReason.trim() : "";
  return reason || "This worker's GPU could not be initialized, so it cannot run jobs.";
}

function isGpuWorker(worker) {
  // Queue resource cards are for live GPU capacity; CPU utility workers stay out of this panel.
  return worker.gpuId && worker.gpuId !== "cpu" && Array.isArray(worker.capabilities) && worker.capabilities.includes("gpu");
}

function formatMemory(mb) {
  if (!Number.isFinite(mb)) {
    return "Unknown";
  }
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }
  return `${Math.round(mb)} MB`;
}

function boundedPercent(value) {
  if (!Number.isFinite(value)) {
    return null;
  }
  return Math.min(100, Math.max(0, value));
}

function memoryUsagePercent(utilization) {
  const total = Number(utilization?.memoryTotalMb);
  const used = Number(utilization?.memoryUsedMb);
  if (!Number.isFinite(total) || total <= 0 || !Number.isFinite(used)) {
    return null;
  }
  return boundedPercent((used / total) * 100);
}

function utilizationLabel(value) {
  return Number.isFinite(value) ? `${Math.round(value)}%` : "Unknown";
}

function WorkerCard({ worker }) {
  const utilization = worker.utilization ?? {};
  const memoryPercent = memoryUsagePercent(utilization);
  const loadPercent = boundedPercent(Number(utilization.gpuLoadPercent));
  const freeMb = Number(utilization.memoryFreeMb);
  const usedMb = Number(utilization.memoryUsedMb);
  const totalMb = Number(utilization.memoryTotalMb);
  const healthReason = workerHealthReason(worker);
  return (
    <div className="worker-card">
      <div className="worker-card-header">
        <strong>{worker.gpuName ?? `GPU ${worker.gpuId}`}</strong>
        <span>{workerStatusLine(worker)}</span>
      </div>
      {/* sc-16260: the host-side remedy, on the card for the worker it applies to. `pre-wrap`
          because the reason is the probe's two-part message — remedy, blank line, raw CUDA error
          — and collapsing that runs them together (the same fix sc-16247 made on the setup
          screen and the queue card). */}
      {healthReason === null ? null : (
        <p className="worker-card-health" role="status" style={{ whiteSpace: "pre-wrap" }}>
          {healthReason}
        </p>
      )}
      <div className="worker-stat-grid">
        <span>
          <small>Available</small>
          <strong>{formatMemory(freeMb)}</strong>
        </span>
        <span>
          <small>Memory</small>
          <strong>{Number.isFinite(usedMb) && Number.isFinite(totalMb) ? `${formatMemory(usedMb)} / ${formatMemory(totalMb)}` : "Unknown"}</strong>
        </span>
        <span>
          <small>Load</small>
          <strong>{utilizationLabel(loadPercent)}</strong>
        </span>
      </div>
      {memoryPercent === null ? null : (
        <div className="worker-meter" aria-label={`GPU memory usage ${utilizationLabel(memoryPercent)}`}>
          <span style={{ width: `${memoryPercent}%` }} />
        </div>
      )}
      {loadPercent === null ? null : (
        <div className="worker-meter gpu-load" aria-label={`GPU load ${utilizationLabel(loadPercent)}`}>
          <span style={{ width: `${loadPercent}%` }} />
        </div>
      )}
      <small>{worker.loadedModels?.length ? `Warm: ${worker.loadedModels.join(", ")}` : "No warm model"}</small>
    </div>
  );
}

export function QueueScreen() {
  const [jobPrompt, setJobPrompt] = useState("Placeholder generation");
  const [selectedJobIds, setSelectedJobIds] = useState(() => new Set());
  const [prioritizing, setPrioritizing] = useState(false);
  const {
    activeProject,
    assets = [],
    cancelPendingJobs,
    clearCompletedJobs,
    clearJob,
    createPlaceholderJob,
    filteredJobs,
    gpuOptions,
    jobAction,
    jobs = filteredJobs,
    prioritizeJobs,
    projectFilter,
    projects,
    requestedGpu,
    setProjectFilter,
    setPreviewAsset,
    setRequestedGpu,
    visibleWorkers,
  } = useAppContext();
  const createJob = (event) => createPlaceholderJob(event, jobPrompt);
  const workers = visibleWorkers;
  // Prefer the shared index from context (sc-2082); fall back for legacy
  // contexts that may not yet expose it (test harnesses, etc.).
  const gpuWorkers = useMemo(() => workers.filter(isGpuWorker), [workers]);
  // "Clear completed" (issue #1556): how many of the jobs in view are terminal
  // (completed / failed / canceled / interrupted) and thus clearable. Gates the
  // button so it disables when there's nothing to clear.
  const completedCount = useMemo(
    () => filteredJobs.filter((job) => terminalStatuses.has(job.status)).length,
    [filteredJobs],
  );
  // "Cancel pending" (sc-13448): how many of the jobs in view are pending
  // (queued / pending_caption) and thus bulk-cancelable. Gates the button so it
  // disables when there's nothing to cancel.
  const pendingCount = useMemo(
    () => filteredJobs.filter((job) => pendingStatuses.has(job.status)).length,
    [filteredJobs],
  );
  const orderedJobs = useMemo(
    () => [...filteredJobs].sort(queueDisplayOrder),
    [filteredJobs],
  );
  const selectableJobIds = useMemo(
    () => new Set(orderedJobs.filter((job) => pendingStatuses.has(job.status)).map((job) => job.id)),
    [orderedJobs],
  );

  // A selected job can be claimed or removed by an SSE update while the operator is deciding.
  // Prune it locally as soon as it stops being pending; the server repeats the same guard under
  // the claim transaction for the final race.
  useEffect(() => {
    setSelectedJobIds((current) => {
      const next = new Set([...current].filter((jobId) => selectableJobIds.has(jobId)));
      return next.size === current.size ? current : next;
    });
  }, [selectableJobIds]);

  const toggleJobSelection = (jobId) => {
    setSelectedJobIds((current) => {
      const next = new Set(current);
      if (next.has(jobId)) next.delete(jobId);
      else next.add(jobId);
      return next;
    });
  };
  const moveSelectedToTop = async () => {
    if (!prioritizeJobs || selectedJobIds.size === 0 || prioritizing) return;
    const jobIds = orderedJobs
      .filter((job) => pendingStatuses.has(job.status) && selectedJobIds.has(job.id))
      .map((job) => job.id);
    if (!jobIds.length) return;
    setPrioritizing(true);
    try {
      const succeeded = await prioritizeJobs(jobIds);
      if (succeeded) setSelectedJobIds(new Set());
    } finally {
      setPrioritizing(false);
    }
  };
  return (
    <section className="page-frame queue-surface">
      <WorkPanel
        eyebrow="Add a job"
        hint="Queue a prompt to a GPU — or Auto to route to the first free worker."
      >
        <form className="job-composer" onSubmit={createJob}>
          <label htmlFor="queue-job-prompt">Prompt</label>
          <input id="queue-job-prompt" onChange={(event) => setJobPrompt(event.target.value)} value={jobPrompt} />
          <label htmlFor="queue-gpu">GPU</label>
          <select id="queue-gpu" onChange={(event) => setRequestedGpu(event.target.value)} value={requestedGpu}>
            {gpuOptions.map((gpu) => (
              <option key={gpu} value={gpu}>
                {gpu === "auto" ? "Auto" : gpu}
              </option>
            ))}
          </select>
          <button disabled={!activeProject} type="submit">
            Add job
          </button>
        </form>
        <div className="work-panel-divider" />
        <div className="queue-tools">
        <label htmlFor="project-filter">Project</label>
        <select id="project-filter" onChange={(event) => setProjectFilter(event.target.value)} value={projectFilter}>
          <option value="all">All projects</option>
          {projects.map((project) => (
            <option key={project.id} value={project.id}>
              {project.name}
            </option>
          ))}
        </select>
        {/* Cancel all pending items in the queue (sc-13448). Scoped to the current
            project filter; disabled when nothing in view is pending. Cancels only
            not-yet-started (queued / pending_caption) jobs — running jobs are left
            to the per-card Cancel so the owning worker acknowledges. */}
        <button
          className="queue-cancel-pending"
          disabled={pendingCount === 0 || !cancelPendingJobs}
          onClick={() => cancelPendingJobs?.(projectFilter)}
          title="Cancel all pending (not-yet-started) jobs in the queue"
          type="button"
        >
          Cancel pending{pendingCount > 0 ? ` (${pendingCount})` : ""}
        </button>
        {/* Clear completed items from the queue (issue #1556). Scoped to the
            current project filter; disabled when nothing in view is terminal. */}
        <button
          className="queue-clear-completed"
          disabled={completedCount === 0 || !clearCompletedJobs}
          onClick={() => clearCompletedJobs?.(projectFilter)}
          title="Remove completed, failed, and canceled jobs from the queue"
          type="button"
        >
          Clear completed{completedCount > 0 ? ` (${completedCount})` : ""}
        </button>
        </div>
      </WorkPanel>

      <div className="worker-grid">
        {gpuWorkers.length === 0 ? (
          <div className="worker-card">
            <strong>No GPU workers registered</strong>
            <span>Start a GPU worker to claim generation jobs.</span>
          </div>
        ) : (
          gpuWorkers.map((worker) => <WorkerCard key={worker.id} worker={worker} />)
        )}
      </div>

      <div className="queue-priority-toolbar">
        <span>Prompt refinement moves to the front automatically. Select waiting jobs to prioritize them.</span>
        <button
          className="queue-move-to-top"
          disabled={selectedJobIds.size === 0 || prioritizing || !prioritizeJobs}
          onClick={moveSelectedToTop}
          type="button"
        >
          {prioritizing ? "Moving…" : `Move to top${selectedJobIds.size ? ` (${selectedJobIds.size})` : ""}`}
        </button>
      </div>

      <div className="job-list">
        {orderedJobs.length === 0 ? (
          <div className="empty-panel">No jobs in this view</div>
        ) : (
          orderedJobs.map((job) => {
            const message = jobWaitingMessage(job, workers, jobs);
            const variant = thumbnailVariantForJob(job);
            const thumbnails = variant === "hidden" ? [] : resolveJobAssets(job, assets);
            // Inject the queue's context-aware waiting/error message via job.message
            // so the shared WorkerProgressCard surface it without per-screen plumbing.
            const enrichedJob = message && message !== job.message ? { ...job, message } : job;
            return (
              <div
                className={`queue-job-entry${selectedJobIds.has(job.id) ? " selected" : ""}`}
                key={job.id}
              >
                {pendingStatuses.has(job.status) ? (
                  <div className="queue-job-selection">
                    <label>
                      <input
                        aria-label={`Select ${formatJobType(job.type)} job ${job.id} to move to the top`}
                        checked={selectedJobIds.has(job.id)}
                        onChange={() => toggleJobSelection(job.id)}
                        type="checkbox"
                      />
                      <span>Select</span>
                    </label>
                    {queueRank(job) > 0 ? <span className="queue-priority-badge">Priority</span> : null}
                  </div>
                ) : null}
                <WorkerProgressCard
                  job={enrichedJob}
                  thumbnailsVariant={variant}
                  thumbnailAssets={thumbnails}
                  onThumbnailClick={setPreviewAsset ? (asset) => setPreviewAsset(asset, thumbnails) : undefined}
                  onCancel={(j) => jobAction(j, "cancel")}
                  onClear={clearJob ? (j) => clearJob(j) : undefined}
                  onRetry={(j, payload) => jobAction(j, "retry", { body: payload ?? {} })}
                  onFreshRetry={(j, payload) => jobAction(j, "retry", { body: payload ?? {} })}
                  onDuplicate={(j) => jobAction(j, "duplicate")}
                  hideOpenQueue
                />
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}

// Variants per job type: asset-producing jobs get the compact small-row of
// thumbnails; caption / import / prompt-refine jobs hide thumbnails per the
// design spec (docs/design/worker-progress-card.md).
function thumbnailVariantForJob(job) {
  switch (job?.type) {
    case "training_caption":
    case "model_download":
    case "model_import":
    case "model_convert":
    case "lora_import":
    case "prompt_refine":
      return "hidden";
    default:
      return "small-row";
  }
}

// Resolve a job's produced asset records against the live catalog. Generic over
// image/video so the queue's small-row works for both (sc-8853): type-agnostic
// (no media-type filter), catalog order for the generationSetId fallback.
function resolveJobAssets(job, assets) {
  return resolveJobResultAssets(job, assets);
}
