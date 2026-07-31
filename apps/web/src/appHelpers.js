// sc-8854 (F-052): pure module-level helpers extracted from App.jsx. These were already
// file-local functions with no closure over component state — worker/job classification,
// SSE parsing, notice mapping, job-list merge/cap, the local-job stack builder, and the
// persisted theme/accent readers. Moving them out of the ~2,650-line App god component
// shrinks its top-of-file surface and makes each independently unit-testable
// (appHelpers.test.js) without mounting App. Behavior is unchanged — this is a move, not a
// rewrite.
import { terminalStatuses } from "./constants.js";
import {
  acceptsJobUpdate,
  capTerminalJobs,
  jobFreshnessMs,
  sortNewest,
  sortOldest,
} from "./sorters.js";
import { DEFAULT_ACCENT, isAccentId } from "./accents.js";

/// A worker that is PRESENT — i.e. still heartbeating. This drives what gets RENDERED
/// (`visibleWorkers` in App.jsx), so an `unhealthy` worker deliberately counts as active
/// (sc-16260): it is registered, alive, and carrying the one message that explains why the queue
/// is stalled. Filtering it here would delete its card and leave the Queue screen claiming "No GPU
/// workers registered" about a worker that is very much registered.
///
/// "Present" is NOT "can take work" — for that, see `canWorkerTakeWork`.
export function isActiveWorker(worker) {
  return worker.status !== "offline";
}

/// A worker that can actually be given work: present AND not reporting an unusable accelerator
/// (sc-16260). Capability-readiness gates use this rather than [`isActiveWorker`], so a workflow
/// is never reported ready on a machine whose GPU cannot be initialized. Mirrors the API's
/// `person_readiness_from_workers`, which excludes the same two statuses.
export function canWorkerTakeWork(worker) {
  return worker.status !== "offline" && worker.status !== "unhealthy";
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

/// sc-15036: a completed `lora_train` job produces EITHER a LoRA/LoKr adapter or a FULL base
/// checkpoint, and each registrar reports under its own result key (`loraRegistered` /
/// `baseCheckpointRegistered`) — only one of which is present on any given run.
///
/// Returns the failure message when THIS run's registration failed, else `null`. Reading only
/// `loraRegistered === false` (what the notice branch did before) meant a failed base-checkpoint
/// registration — where `loraRegistered` is `undefined` — fell into the success path, so the run
/// reported success while its own result said the ~8 GB checkpoint had not been registered.
///
/// Strict `=== false` on both keys, deliberately: an ABSENT key means "that registrar did not
/// claim this job", which is the normal case for the other kind and must never read as a failure.
export function trainingRegistrationFailure(result) {
  const failed = result?.loraRegistered === false || result?.baseCheckpointRegistered === false;
  if (!failed) {
    return null;
  }
  return (
    result?.loraRegistrationError ??
    result?.baseCheckpointRegistrationError ??
    "Completed training but could not register the result."
  );
}

export function isImageGenerationJob(job) {
  return ["image_generate", "image_edit"].includes(job.type);
}

export function isVideoGenerationJob(job) {
  return ["video_generate", "video_extend", "video_bridge"].includes(job.type);
}

// SceneWorks Audio Studio (epic 13400 / sc-13404): the audio-generation job type, the audio twin of
// isVideoGenerationJob. Powers the `audio` local-job lane (rememberLocalGenerationJob('audio', job)).
export function isAudioGenerationJob(job) {
  return job.type === "audio_generate";
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

export { jobFreshnessMs };

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

// Reconnect snapshots are authoritative for active jobs: the server payload
// includes its uncapped active set, so a locally cached active row absent from
// the snapshot has transitioned or been cleared while the stream was down.
// Terminal client-only history remains retained because the server intentionally
// caps its recent-job snapshot. Duplicate rows still resolve by updatedAt so an
// in-flight live event cannot be regressed by an older snapshot.
export function reconcileAuthoritativeJobs(currentJobs, serverJobs, clearedJobIds = []) {
  const cleared = new Set(clearedJobIds);
  // Tombstones win even if a server-side composition bug includes the same row
  // in both collections. Never let a soft-hidden job re-enter local state.
  const merged = new Map(
    serverJobs
      .filter((job) => !cleared.has(job.id))
      .map((job) => [job.id, job]),
  );
  for (const current of currentJobs) {
    if (cleared.has(current.id)) {
      continue;
    }
    const server = merged.get(current.id);
    if (server) {
      if (!acceptsJobUpdate([current], server)) {
        merged.set(current.id, current);
      }
    } else if (terminalStatuses.has(current.status)) {
      merged.set(current.id, current);
    }
  }
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
