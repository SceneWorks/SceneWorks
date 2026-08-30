// Batch operations across multiple assets (sc-6112, Workstream F of epic 6087).
// Pure orchestration: turn a multi-asset selection + one op (upscale / detail / edit)
// + shared params into one job body PER asset, and summarize the fan-out's progress
// from the global jobs feed. React/DOM-free (reuses the konva-free imageJobs.js
// builders) so the fan-out math is unit-tested in isolation and the Library bundle
// never pulls in the editor / react-konva.

import { terminalStatuses } from "./constants.js";
import { buildDetailJobBody, buildEditJobBody, buildUpscaleJobBody } from "./imageJobs.js";

// The three batch-capable ops + the endpoint each posts to (mirrors the editor:
// upscale/detail are generic jobs, edit is the image-jobs route). `needsPrompt`
// flags the op whose shared params include a text prompt.
export const BATCH_OPS = [
  { key: "upscale", label: "Upscale", endpoint: "/api/v1/jobs", needsPrompt: false },
  { key: "detail", label: "Detail enhance", endpoint: "/api/v1/jobs", needsPrompt: false },
  { key: "edit", label: "AI edit", endpoint: "/api/v1/image/jobs", needsPrompt: true },
];

export function batchOpByKey(key) {
  return BATCH_OPS.find((op) => op.key === key) ?? null;
}

// Assets a raster op can run on — images (by type or image/* mime), never clips.
export function batchEligibleAssets(assets) {
  return (assets ?? []).filter((asset) => {
    if (asset?.type === "video") return false;
    const mime = asset?.file?.mimeType ?? "";
    return asset?.type === "image" || mime.startsWith("image/");
  });
}

// Build the `{ endpoint, body }` for one asset under the chosen op + shared params.
// `dims` (the asset's native width/height) is REQUIRED for edit — the worker fits the
// source to width×height — and ignored by upscale/detail. Throws on an unknown op or
// missing edit dims so a bad fan-out fails loudly rather than posting a malformed job.
export function buildBatchJob({ op, asset, params = {}, project, requestedGpu, dims = null }) {
  const sourceAssetId = asset.id;
  const displayName = asset.displayName ?? asset.id;
  if (op === "upscale") {
    return {
      endpoint: "/api/v1/jobs",
      body: buildUpscaleJobBody({
        project,
        requestedGpu,
        sourceAssetId,
        factor: params.factor,
        engine: params.engine,
        displayName,
        softness: params.softness,
      }),
    };
  }
  if (op === "detail") {
    return {
      endpoint: "/api/v1/jobs",
      body: buildDetailJobBody({
        project,
        requestedGpu,
        sourceAssetId,
        model: params.model,
        strength: params.strength,
        cnScale: params.cnScale,
        displayName,
      }),
    };
  }
  if (op === "edit") {
    if (!dims?.width || !dims?.height) {
      throw new Error("Batch edit needs the source image dimensions.");
    }
    return {
      endpoint: "/api/v1/image/jobs",
      body: buildEditJobBody({
        project,
        requestedGpu,
        sourceAssetId,
        model: params.model,
        prompt: params.prompt,
        seed: params.seed,
        width: dims.width,
        height: dims.height,
        // Same-size edit (no canvas extend) — each image is edited at its native size.
        fitMode: "crop",
      }),
    };
  }
  throw new Error(`Unknown batch op: ${op}`);
}

// One batch item's status, derived from the capped global jobs feed. `observedJobs`
// is owned by the batch UI: once a job has surfaced, active jobs cannot disappear
// from the feed, so a later absence means the job reached a terminal state and was
// evicted from retained history. Preserve an observed terminal outcome; otherwise
// treat a vanished active job as completed instead of queued forever.
export function batchItemStatus(jobId, jobs, observedJobs) {
  if (!jobId) return "queued";
  const job = (jobs ?? []).find((item) => item.id === jobId);
  if (!job) {
    const observed = observedJobs?.get(jobId);
    return observed === "failed" || observed === "completed"
      ? observed
      : observedJobs?.has(jobId)
        ? "completed"
        : "queued";
  }
  if (job.status === "completed") return "completed";
  if (terminalStatuses.has(job.status)) return "failed"; // failed / canceled / interrupted
  if (job.status === "running") return "running";
  return "queued";
}

// Resolve the status of the batch item shape. Sequential fan-out starts with
// explicit pending placeholders; once submission settles, a missing job id is
// a failure because every successful response must provide one.
export function batchItemStatusForItem(item, jobs, observedJobs) {
  if (!item?.jobId) return item?.pending ? "queued" : "failed";
  return batchItemStatus(item.jobId, jobs, observedJobs);
}

// The completed result asset for a batch item (the API exposes persisted worker asset writes on
// `result.assets[0]`), or null while pending / on failure.
export function batchItemResultAsset(jobId, jobs) {
  const job = (jobs ?? []).find((item) => item.id === jobId);
  if (!job || job.status !== "completed") return null;
  return job.result?.assets?.[0] ?? null;
}

// Aggregate a fan-out's progress. `items` = [{ assetId, jobId }]; returns the per-status
// tallies plus `done` (terminal) and `allDone`. Items with no jobId (submission failed)
// count as failed so the aggregate never claims more progress than really happened.
export function summarizeBatchProgress(items, jobs, observedJobs) {
  const summary = { total: items?.length ?? 0, queued: 0, running: 0, completed: 0, failed: 0 };
  for (const item of items ?? []) {
    summary[batchItemStatusForItem(item, jobs, observedJobs)] += 1;
  }
  summary.done = summary.completed + summary.failed;
  summary.allDone = summary.total > 0 && summary.done === summary.total;
  return summary;
}

// Move terminal active jobs into the bounded prompt-batch totals. The active-id list is
// deliberately small (the Studio applies backpressure before it grows without bound), while
// completed and failed are scalar counters. Returns the original run when nothing settled.
export function settlePromptBatchRun(run, jobs, observedJobs) {
  if (!run?.activeJobIds?.length) return run;
  let completed = run.completed ?? 0;
  let failed = run.failed ?? 0;
  const activeJobIds = [];
  for (const jobId of run.activeJobIds) {
    const status = batchItemStatus(jobId, jobs, observedJobs);
    if (status === "completed") {
      completed += 1;
      observedJobs?.delete(jobId);
    } else if (status === "failed") {
      failed += 1;
      observedJobs?.delete(jobId);
    } else {
      activeJobIds.push(jobId);
    }
  }
  if (activeJobIds.length === run.activeJobIds.length) return run;
  return { ...run, completed, failed, activeJobIds };
}

// Progress for a prompt-batch run (sc-9980). It holds scalar totals plus only the bounded
// currently-active job ids, never one entry per resolved prompt. `submitted` includes posts
// that failed before returning a job id; `pending` is the not-yet-submitted suffix.
export function summarizeBatchRun(run, jobs, observedJobs) {
  const total = Math.max(run?.total ?? 0, 0);
  const submitted = Math.min(Math.max(run?.submitted ?? 0, 0), total);
  const summary = {
    total,
    pending: total - submitted,
    queued: 0,
    running: 0,
    completed: run?.completed ?? 0,
    failed: run?.failed ?? 0,
  };
  for (const jobId of run?.activeJobIds ?? []) {
    summary[batchItemStatus(jobId, jobs, observedJobs)] += 1;
  }
  summary.done = summary.completed + summary.failed;
  summary.active = summary.queued + summary.running;
  summary.allDone = summary.total > 0 && summary.done === summary.total;
  return summary;
}
