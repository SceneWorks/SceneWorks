// sc-8854 (F-052): pure module-level helpers extracted from App.jsx. These were already
// file-local functions with no closure over component state — worker/job classification,
// SSE parsing, notice mapping, job-list merge/cap, the local-job stack builder, and the
// persisted theme/accent readers. Moving them out of the ~2,650-line App god component
// shrinks its top-of-file surface and makes each independently unit-testable
// (appHelpers.test.js) without mounting App. Behavior is unchanged — this is a move, not a
// rewrite.
import { terminalStatuses } from "./constants.js";
import { capTerminalJobs, sortNewest, sortOldest } from "./sorters.js";
import { DEFAULT_ACCENT, isAccentId } from "./accents.js";

export function isActiveWorker(worker) {
  return worker.status !== "offline";
}

export function hasCapability(worker, capability) {
  return Array.isArray(worker.capabilities) && worker.capabilities.includes(capability);
}

export function isPlaceholderOnlyGpuWorker(worker) {
  if (!hasCapability(worker, "gpu")) {
    return false;
  }
  const capabilities = Array.isArray(worker.capabilities) ? worker.capabilities : [];
  return capabilities.every((capability) => ["placeholder", "gpu", "nvidia"].includes(capability));
}

export function isSelectableGpuWorker(worker) {
  return worker.gpuId && worker.gpuId !== "cpu" && hasCapability(worker, "gpu") && !isPlaceholderOnlyGpuWorker(worker);
}

export function failedJobNotice(job) {
  const label = String(job.type ?? "job").replaceAll("_", " ");
  const detail = job.error || job.message || "Failed without additional worker detail.";
  return `${label}: ${detail}`;
}

export function isImageGenerationJob(job) {
  return ["image_generate", "image_edit"].includes(job.type);
}

export function isVideoGenerationJob(job) {
  return ["video_generate", "video_extend", "video_bridge"].includes(job.type);
}

export function isInterleaveJob(job) {
  return job.type === "image_interleave";
}

export function parseSseJson(event, label) {
  try {
    return JSON.parse(event.data);
  } catch (err) {
    console.warn(`Ignoring malformed ${label} SSE event`, err);
    return null;
  }
}

// sc-4198: notice kind for a job-failure banner. LoRA import/train failures get
// their own kind so the matching job's later completion dismisses exactly that
// banner (replacing the old "lora import:"/"lora training:" startsWith protocol);
// everything else is a general error.
export function noticeKindForJob(job) {
  if (job?.type === "lora_import") return "lora-import";
  if (job?.type === "lora_train") return "lora-train";
  return "general";
}

export function jobFreshnessMs(job) {
  const timestamp = job?.updatedAt ?? job?.completedAt ?? job?.canceledAt ?? job?.startedAt ?? job?.createdAt;
  const parsed = Date.parse(timestamp ?? "");
  return Number.isFinite(parsed) ? parsed : 0;
}

export function mergeFreshJobs(currentJobs, serverJobs) {
  const merged = new Map();
  for (const job of serverJobs) {
    merged.set(job.id, job);
  }
  for (const current of currentJobs) {
    const server = merged.get(current.id);
    if (!server || jobFreshnessMs(current) > jobFreshnessMs(server)) {
      merged.set(current.id, current);
    }
  }
  // sc-8860 (F-058): this deliberately keeps client-side entries the server no
  // longer returns, so without a cap a long session grows unbounded. Cap the
  // retained terminal-job tail (active jobs are never dropped) so a refresh can't
  // monotonically grow `jobs`.
  return capTerminalJobs([...merged.values()].sort(sortNewest));
}

export function generatedResultAssetCount(job) {
  if (Array.isArray(job.result?.assetIds)) {
    return job.result.assetIds.length;
  }
  if (Array.isArray(job.result?.assets)) {
    return job.result.assets.length;
  }
  return 0;
}

// Studios stack every running and queued run (plus the most recent finished run
// until its successor starts), so a new submission no longer evicts the prior
// progress card. Capped so a long session can't grow the visible stack unbounded.
export const localJobStackLimit = 25;

// Build a studio's local-job stack: the runs it explicitly remembered plus any
// still-active generation jobs for the open project, de-duped and ordered
// oldest-first (running run on top, queued runs following in execution order),
// keeping only the most recent `localJobStackLimit` entries.
export function buildLocalJobStack(rememberedIds, jobs, activeProjectId, isGenerationJob) {
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
  return Array.from(byId.values()).sort(sortOldest).slice(-localJobStackLimit);
}

export function readStoredTheme() {
  if (typeof window === "undefined") {
    return "light";
  }
  try {
    const saved = window.localStorage.getItem("sceneworks-theme");
    return saved === "dark" || saved === "light" ? saved : "light";
  } catch {
    return "light";
  }
}

export function readStoredAccent() {
  if (typeof window === "undefined") {
    return DEFAULT_ACCENT;
  }
  try {
    const saved = window.localStorage.getItem("sceneworks-accent");
    return isAccentId(saved) ? saved : DEFAULT_ACCENT;
  } catch {
    return DEFAULT_ACCENT;
  }
}
