import { useEffect, useRef } from "react";
import { iterateBatch, parsePromptResolution } from "../../promptBatch.js";
import { batchItemStatus, recordBatchEnqueueFailure, settlePromptBatchRun, summarizeBatchRun } from "../../batchOps.js";
import { serializeCaption, validateCaption } from "../../ideogramCaption.js";
import { promptBatchSize } from "./samplingControls.js";
import { batchPromptBudgetMessage, batchPromptBudgetOverages } from "../../imageStudioValidation.js";
import { PROMPT_REFINE_MODEL_ID, terminalStatuses } from "../../constants.js";

// Extracted from ImageStudio along its existing concern boundary.
export function useImageBatchRun({
  batchRun,
  activeProject,
  jobs,
  jobAction,
  setBatchRun,
  BATCH_ACTIVE_JOB_LIMIT,
  preflightError,
  setBatchError,
  batchTotal,
  BATCH_RENDER_CAP,
  setBatchConfirmPending,
  batchPrompts,
  batchVariables,
  resolutionOptions,
  composeJobPrompt,
  stylePreviewActive,
  structuredPromptModel,
  batchJobCount,
  BATCH_PREFLIGHT_YIELD_INTERVAL,
  selectedModel,
  buildBatchJobRequest,
  onMagicExpand,
  createImageJob
}) {
  const batchObservedJobsRef = useRef(new Map());

  const batchRunRef = useRef(batchRun);

  const batchSlotWaitersRef = useRef([]);

  const batchOwnerRef = useRef(null);

  const batchProjectRef = useRef(activeProject?.id);

  batchProjectRef.current = activeProject?.id;

  const batchJobsRef = useRef(jobs);

  batchJobsRef.current = jobs;

  const batchJobActionRef = useRef(jobAction);

  batchJobActionRef.current = jobAction;

  useEffect(() => {
    const waiters = batchSlotWaitersRef.current;
    if (batchOwnerRef.current && batchOwnerRef.current.projectId !== activeProject?.id) {
      batchOwnerRef.current = null;
      batchRunRef.current = null;
      setBatchRun(null);
    }
    return () => {
      const owner = batchOwnerRef.current;
      if (!owner) return;
      owner.stopped = true;
      owner.controller.abort();
      owner.disposed = true;
      for (const resolve of waiters.splice(0)) resolve();
      // Project changes and unmount use the same policy as Stop: cancel jobs
      // already enqueued, including a post that resolves after cleanup.
      for (const [id, submittedJob] of owner.jobs) {
        const job = batchJobsRef.current.find((entry) => entry.id === id) ?? submittedJob;
        if (job && !terminalStatuses.has(job.status)) batchJobActionRef.current(job, "cancel");
      }
    };
  }, [activeProject?.id, setBatchRun]);

  function replaceBatchRun(next) {
    batchRunRef.current = next;
    setBatchRun(next);
  }

  function releaseBatchSlotWaiters() {
    const waiters = batchSlotWaitersRef.current.splice(0);
    for (const resolve of waiters) resolve();
  }

  async function waitForBatchSlot(owner) {
    while (!owner.stopped && batchProjectRef.current === owner.projectId) {
      const activeJobIds = batchRunRef.current?.activeJobIds ?? [];
      if (activeJobIds.length < BATCH_ACTIVE_JOB_LIMIT) return;
      await new Promise((resolve) => batchSlotWaitersRef.current.push(resolve));
    }
  }

  // Fan out one image job per resolved prompt (mirrors the asset batch, sc-6112): each
  // posts independently so the worker runs them serially with its between-image cache
  // release, and progress/cancel read the live jobs feed.
  async function runBatch(confirmed = false) {
    if (batchRunRef.current?.submitting || !activeProject) return;
    const error = preflightError("batch");
    if (error) {
      setBatchError(error);
      return;
    }
    // Admission precedes expansion; confirmation preserves the exact image count.
    if (!confirmed && batchTotal > BATCH_RENDER_CAP) {
      setBatchConfirmPending(true);
      return;
    }
    const styledBatchPrompts = () => (function* composedBatchPrompts() {
      for (const { prompt: resolvedPrompt } of iterateBatch(batchPrompts, batchVariables)) {
        const { prompt: cleanPrompt } = parsePromptResolution(resolvedPrompt, resolutionOptions);
        yield composeJobPrompt({ promptToSend: cleanPrompt }).composedPrompt;
      }
    })();
    // An ordinary batch remains eager so its warning retains every actionable item.
    // Only a confirmed high-cardinality product needs the cancelable streaming path below.
    if (stylePreviewActive && !structuredPromptModel && batchJobCount <= BATCH_RENDER_CAP) {
      const promptBudgetOverages = batchPromptBudgetOverages(styledBatchPrompts());
      if (promptBudgetOverages.length) {
        setBatchError(batchPromptBudgetMessage(promptBudgetOverages));
        return;
      }
    }
    setBatchError("");
    setBatchConfirmPending(false);
    const owner = { stopped: false, projectId: activeProject.id, jobs: new Map(), controller: new AbortController() };
    batchOwnerRef.current = owner;
    const stopped = () => owner.stopped || batchProjectRef.current !== owner.projectId || batchOwnerRef.current !== owner;
    batchObservedJobsRef.current.clear();
    // Keep scalar totals plus a bounded in-flight id set. Once the set reaches the normal
    // batch cap, wait for a terminal job before posting the next resolved prompt.
    replaceBatchRun({
      submitting: true,
      total: batchJobCount,
      submitted: 0,
      completed: 0,
      failed: 0,
      activeJobIds: [],
    });
    // Validate styled prompts before any job is admitted. This is deliberately a stream:
    // a high-cardinality all-under-cap Cartesian product used to synchronously exhaust here,
    // before React could paint the run and make Stop usable. The state above establishes the
    // cancel owner first; bounded cooperative yields preserve truthful preflight enforcement
    // without retaining every prompt or letting any job through before validation completes.
    if (stylePreviewActive && !structuredPromptModel && batchJobCount > BATCH_RENDER_CAP) {
      let inspected = 0;
      for (const { prompt: resolvedPrompt } of iterateBatch(batchPrompts, batchVariables)) {
        if (stopped()) {
          const current = batchRunRef.current;
          if (current && !owner.disposed && batchOwnerRef.current === owner && batchProjectRef.current === owner.projectId) replaceBatchRun({ ...current, submitting: false });
          return;
        }
        const { prompt: cleanPrompt } = parsePromptResolution(resolvedPrompt, resolutionOptions);
        const promptBudgetOverages = batchPromptBudgetOverages(
          [composeJobPrompt({ promptToSend: cleanPrompt }).composedPrompt],
          1,
        );
        if (promptBudgetOverages.length) {
          const [{ item: _item, ...budget }] = promptBudgetOverages;
          replaceBatchRun(null);
          setBatchError(batchPromptBudgetMessage([{ item: inspected + 1, ...budget }]));
          return;
        }
        inspected += 1;
        if (batchJobCount > BATCH_RENDER_CAP && inspected % BATCH_PREFLIGHT_YIELD_INTERVAL === 0) {
          // A macrotask, rather than a microtask, gives the browser an event turn to deliver Stop.
          await new Promise((resolve) => window.setTimeout(resolve, 0));
        }
      }
    }
    // sc-25681: a model with a native prompt batch (Iris-3B) packs consecutive same-size prompts
    // into one job of up to `promptBatchSize` prompts; every other model keeps one job per prompt.
    // The job's own `count` (a preset stack may override the studio's), so prompts × count fits.
    const size = promptBatchSize(selectedModel, buildBatchJobRequest("").count);
    function* chunks() {
      let chunk = [];
      for (const entry of iterateBatch(batchPrompts, batchVariables)) {
        // Strip a leading [WxH] directive (sc-10063): the model gets the clean prompt, the job
        // gets that per-prompt resolution.
        const item = parsePromptResolution(entry.prompt, resolutionOptions);
        if (chunk.length >= size || (chunk[0] && JSON.stringify(chunk[0].resolution) !== JSON.stringify(item.resolution))) {
          yield chunk;
          chunk = [];
        }
        chunk.push(item);
      }
      if (chunk.length) yield chunk;
    }
    for (const chunk of chunks()) {
      await waitForBatchSlot(owner);
      if (stopped()) {
        break;
      }
      const [{ prompt: cleanPrompt, resolution }] = chunk;
      const n = chunk.length;
      try {
        let request;
        if (structuredPromptModel) {
          // Structured-caption models (Ideogram 4) reject raw plain text, so auto-expand each
          // resolved prompt into a JSON caption first (sc-9980) — N sequential refine calls.
          // A prompt that fails to expand fails only that item; the rest continue.
          const expanded = await onMagicExpand(cleanPrompt);
          if (stopped()) break;
          if (!validateCaption(expanded).ok) {
            throw new Error("Auto-generated caption was invalid.");
          }
          request = buildBatchJobRequest(cleanPrompt, {
            promptToSend: serializeCaption(expanded),
            sendStructured: true,
            submitCaption: expanded,
            submitBackend: PROMPT_REFINE_MODEL_ID,
            resolution,
          });
        } else {
          request = buildBatchJobRequest(cleanPrompt, { resolution });
          // Each prompt composed exactly as its own job would be; `prompt` stays prompts[0].
          if (n > 1) request.prompts = chunk.map((item) => buildBatchJobRequest(item.prompt, { resolution }).prompt);
        }
        if (stopped()) break;
        const result = createImageJob.withResult
          ? await createImageJob.withResult([request], { shouldContinue: () => !stopped(), signal: owner.controller.signal })
          : { job: await createImageJob(request) };
        if (result.cancelled) break;
        const { job } = result;
        const failure = result.error || "Job creation returned no job ID.";
        if (stopped()) {
          if (job?.id) jobAction(job, "cancel");
          if (!owner.disposed && batchOwnerRef.current === owner && batchProjectRef.current === owner.projectId) {
            const current = batchRunRef.current;
            if (current) {
              if (job?.id) {
                owner.jobs.set(job.id, job);
                replaceBatchRun({ ...current, submitted: current.submitted + n,
                  activeJobIds: [...current.activeJobIds, job.id], sizes: { ...current.sizes, [job.id]: n } });
              } else {
                replaceBatchRun(recordBatchEnqueueFailure(current, n, failure));
              }
            }
          }
          break;
        }
        if (job?.id) owner.jobs.set(job.id, job);
        const current = batchRunRef.current;
        if (!current) break;
        replaceBatchRun(
          job?.id
            ? {
                ...current,
                submitted: current.submitted + n,
                activeJobIds: [...current.activeJobIds, job.id],
                sizes: { ...current.sizes, [job.id]: n },
              }
            : recordBatchEnqueueFailure(current, n, failure),
        );
        if (result.stop) break;
      } catch (error) {
        if (owner.disposed || batchOwnerRef.current !== owner || batchProjectRef.current !== owner.projectId) break;
        const current = batchRunRef.current;
        if (!current) break;
        replaceBatchRun(recordBatchEnqueueFailure(current, n, error?.message || (typeof error === "string" ? error : "Could not enqueue image job.")));
        if (stopped()) break;
      }
    }
    const current = batchRunRef.current;
    if (current && !stopped()) replaceBatchRun({ ...current, submitting: false });
    else if (current && !owner.disposed && batchOwnerRef.current === owner && batchProjectRef.current === owner.projectId) replaceBatchRun({ ...current, submitting: false });
  }

  // Stop the enqueue loop (if still running) and cancel every still-pending job in the run;
  // completed/failed items are left as-is.
  function cancelBatchRun() {
    if (batchOwnerRef.current) {
      batchOwnerRef.current.stopped = true;
      batchOwnerRef.current.controller.abort();
    }
    releaseBatchSlotWaiters();
    const current = batchRunRef.current;
    if (!current) {
      return;
    }
    for (const jobId of current.activeJobIds) {
      const status = batchItemStatus(jobId, jobs, batchObservedJobsRef.current);
      if (status !== "queued" && status !== "running") {
        continue;
      }
      const job = jobs.find((entry) => entry.id === jobId) ?? batchOwnerRef.current?.jobs.get(jobId);
      if (job) {
        jobAction(job, "cancel");
      }
    }
  }

  const batchRunProgress = batchRun
    ? summarizeBatchRun(batchRun, jobs, batchObservedJobsRef.current)
    : null;

  useEffect(() => {
    const current = batchRunRef.current;
    if (!current) return;
    const batchJobIds = new Set(current.activeJobIds);
    for (const job of jobs) {
      if (batchJobIds.has(job.id)) {
        const status =
          job.status === "completed"
            ? "completed"
            : terminalStatuses.has(job.status)
              ? "failed"
              : "observed";
        batchObservedJobsRef.current.set(job.id, status);
      }
    }
    const settled = settlePromptBatchRun(current, jobs, batchObservedJobsRef.current);
    if (settled !== current) {
      for (const id of current.activeJobIds) {
        if (!settled.activeJobIds.includes(id)) batchOwnerRef.current?.jobs.delete(id);
      }
      batchRunRef.current = settled;
      setBatchRun(settled);
      releaseBatchSlotWaiters();
    }
  }, [batchRun, jobs, setBatchRun]);
  return { replaceBatchRun, runBatch, cancelBatchRun, batchRunProgress };
}
