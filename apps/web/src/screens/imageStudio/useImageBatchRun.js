import { useImageBatchController } from "../../hooks/useImageBatchController.js";
export { useImageBatchController };
import { iterateBatch, parsePromptResolution } from "../../promptBatch.js";
import { BATCH_SUBMISSION_CONCURRENCY, batchItemStatus, recordBatchEnqueueFailure, settlePromptBatchRun, summarizeBatchRun } from "../../batchOps.js";
import { createJobSubmissionGroup } from "../../createJob.js";
import { serializeCaption, validateCaption } from "../../ideogramCaption.js";
import { promptBatchSize } from "./samplingControls.js";
import { PROMPT_REFINE_MODEL_ID, terminalStatuses } from "../../constants.js";
import { MAX_VALIDATION_BATCH, validatePortableBatch } from "../../portableValidation.js";

// This factory receives only the publisher, so an asynchronous run cannot pin
// a retired React render's jobs, catalogs, previews, or component callbacks.
export function createImageBatchRuntime(setBatchRun) {
  const ref = (current) => ({ current });
  const batchObservedJobsRef = ref(new Map());

  const batchRunRef = ref(null);

  const batchSlotWaitersRef = ref([]);

  const batchOwnerRef = ref(null);

  const batchProjectRef = ref(null);

  const batchJobsRef = ref([]);

  const batchJobActionRef = ref(null);

  function replaceBatchRun(next) {
    batchRunRef.current = next;
    setBatchRun(next);
  }

  function releaseBatchSlotWaiters() {
    const waiters = batchSlotWaitersRef.current.splice(0);
    for (const resolve of waiters) resolve();
  }

  async function waitForBatchSlot(owner, activeJobLimit) {
    while (!owner.stopped && !owner.enqueueStopped && batchProjectRef.current === owner.projectId && batchOwnerRef.current === owner) {
      const activeJobIds = batchRunRef.current?.activeJobIds ?? [];
      if (activeJobIds.length + owner.submissions.size < activeJobLimit) return;
      await new Promise((resolve) => batchSlotWaitersRef.current.push(resolve));
    }
  }

  // Expand every resolved prompt with MiniMax-H3-Image's expander, one job after another. A
  // prompt whose expansion fails is recorded as a failed batch item (with the reason) and not
  // rendered; Stop ends the phase and queues nothing. Returns the items to render.
  async function expandBatchItems(entries, owner, stopped, batchExpansion) {
    const { expandPrompt, copy, modelId, referenceAssetIds, useSizes, proposedResolution, failureMessage } = batchExpansion;
    const ready = [];
    const progress = (done) => {
      const current = batchRunRef.current;
      if (current) replaceBatchRun({ ...current, expanding: { done, total: entries.length } });
    };
    progress(0);
    for (let index = 0; index < entries.length; index += 1) {
      if (stopped()) return [];
      const item = entries[index];
      if (!item.prompt.trim()) {
        ready.push(item);
        progress(index + 1);
        continue;
      }
      try {
        const result = await expandPrompt({
          prompt: item.prompt.trim(),
          modelId,
          projectId: owner.projectId,
          sourceAssetIds: referenceAssetIds,
          signal: owner.controller.signal,
        });
        if (stopped()) return [];
        const expanded = String(result?.refinedPrompt ?? "").trim();
        if (!expanded) throw new Error("The prompt expander returned an empty prompt.");
        // The proposed size is opt-in, and a prompt's own [size] always wins.
        const size = useSizes && !item.resolution ? /^(\d+)x(\d+)$/.exec(proposedResolution(result?.rewriteSuggestion) || "") : null;
        ready.push({
          prompt: expanded,
          resolution: size ? { width: Number(size[1]), height: Number(size[2]) } : item.resolution,
        });
      } catch (error) {
        if (stopped() || error?.name === "AbortError") return [];
        const current = batchRunRef.current;
        if (!current) return [];
        replaceBatchRun(recordBatchEnqueueFailure(current, 1, copy.batch.failed(index + 1, failureMessage(error))));
      }
      progress(index + 1);
    }
    const current = batchRunRef.current;
    if (current) replaceBatchRun({ ...current, expanding: null });
    return ready;
  }

  // Fan out one image job per resolved prompt (mirrors the asset batch, sc-6112): each
  // posts independently so the worker runs them serially with its between-image cache
  // release, and progress/cancel read the live jobs feed.
  async function runBatch({
    BATCH_ACTIVE_JOB_LIMIT, batchPrompts, batchVariables, resolutionOptions, sizeEnvelope = null,
    structuredPromptModel, batchJobCount, selectedModel, token = "",
    buildBatchJobRequest, onMagicExpand, batchExpansion = null, createImageJob,
  }) {
    if (batchRunRef.current?.submitting || !batchProjectRef.current) return;
    const owner = { stopped: false, enqueueStopped: false, projectId: batchProjectRef.current,
      jobs: new Map(), submissions: new Set(), submissionGroup: createJobSubmissionGroup(), controller: new AbortController() };
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
    // sc-25681: a model with a native prompt batch (Iris-3B) packs consecutive same-size prompts
    // into one job of up to `promptBatchSize` prompts; every other model keeps one job per prompt.
    // The job's own `count` (a preset stack may override the studio's), so prompts × count fits.
    const size = promptBatchSize(selectedModel, buildBatchJobRequest("").count);
    function* resolvedItems() {
      for (const entry of iterateBatch(batchPrompts, batchVariables)) {
        // Strip a leading [WxH] directive (sc-10063): the model gets the clean prompt, the job
        // gets that per-prompt resolution.
        yield parsePromptResolution(entry.prompt, resolutionOptions, sizeEnvelope);
      }
    }
    // Prompt expansion runs to the end BEFORE the first image job is queued, so the expander and
    // the image model each load once rather than alternating per prompt.
    const items = batchExpansion ? await expandBatchItems([...resolvedItems()], owner, stopped, batchExpansion) : resolvedItems();
    function* chunks() {
      let chunk = [];
      for (const item of items) {
        if (chunk.length >= size || (chunk[0] && JSON.stringify(chunk[0].resolution) !== JSON.stringify(item.resolution))) {
          yield chunk;
          chunk = [];
        }
        chunk.push(item);
      }
      if (chunk.length) yield chunk;
    }
    const chunkIterator = chunks();
    let nextChunk = chunkIterator.next();
    // All direct-prompt models share bounded validation windows. Keep structured-caption
    // refinement just-in-time, so stopping or a refusal never rewrites later prompts.
    async function prepareValidationBatch() {
      const prepared = [];
      const limit = structuredPromptModel ? 1 : MAX_VALIDATION_BATCH;
      for (let index = 0; index < limit && !stopped(); index += 1) {
        if (nextChunk.done) break;
        const chunk = nextChunk.value;
        nextChunk = chunkIterator.next();
        const [{ prompt: cleanPrompt, resolution }] = chunk;
        const entry = { n: chunk.length };
        prepared.push(entry);
        try {
          if (structuredPromptModel) {
            // Structured-caption models need a JSON caption before validation (sc-9980).
            // A prompt that fails to expand fails only that item; the rest continue.
            const expanded = await onMagicExpand(cleanPrompt);
            if (stopped()) return [];
            if (!validateCaption(expanded).ok) {
              throw new Error("Auto-generated caption was invalid.");
            }
            entry.request = buildBatchJobRequest(cleanPrompt, {
              promptToSend: serializeCaption(expanded),
              sendStructured: true,
              submitCaption: expanded,
              submitBackend: PROMPT_REFINE_MODEL_ID,
              resolution,
            });
          } else {
            entry.request = buildBatchJobRequest(cleanPrompt, { resolution, appearancePromptBatch: entry.n > 1 ? chunk.map((item) => item.prompt) : undefined });
            // Each prompt composed exactly as its own job would be; `prompt` stays prompts[0].
            if (entry.n > 1) {
              const requests = chunk.map((item) => buildBatchJobRequest(item.prompt, { resolution }));
              entry.request.prompts = requests.map((item) => item.prompt);
            }
          }
        } catch (error) {
          entry.error = error;
        }
      }
      const ready = prepared.filter((entry) => !("error" in entry));
      if (ready.length && !stopped()) {
        try {
          const results = await validatePortableBatch({
            kind: "image", requests: ready.map(({ request }) => ({ ...request, projectId: owner.projectId })),
            token, signal: owner.controller.signal,
          });
          ready.forEach((entry, index) => { entry.portable = results[index]; });
        } catch (cause) {
          const error = new Error(cause?.message || "Could not validate image jobs.");
          error.portableStop = true;
          ready.forEach((entry) => { entry.error = error; });
        }
      }
      return prepared;
    }
    function stopEnqueue() {
      owner.enqueueStopped = true;
      releaseBatchSlotWaiters();
    }
    async function submitEntry(entry) {
      const { n, request, portable } = entry;
      try {
        if ("error" in entry) throw entry.error;
        if (stopped()) return;
        if (portable.state !== "valid") {
          const error = new Error(portable.state === "invalid"
            ? portable.refusal.detail
            : "The server must check these options. Reconnect to continue.");
          error.portableStop = true;
          throw error;
        }
        const result = createImageJob.withResult
          ? await createImageJob.withResult([request], { shouldContinue: () => !stopped(), signal: owner.controller.signal,
              submissionGroup: owner.submissionGroup })
          : { job: await createImageJob(request) };
        if (result.cancelled) { stopEnqueue(); return; }
        const { job } = result;
        const failure = result.error || "Job creation returned no job ID.";
        if (stopped()) {
          if (job?.id) batchJobActionRef.current(job, "cancel");
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
          return;
        }
        if (job?.id) owner.jobs.set(job.id, job);
        const current = batchRunRef.current;
        if (!current) return;
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
        if (result.stop) stopEnqueue();
      } catch (error) {
        if (owner.disposed || batchOwnerRef.current !== owner || batchProjectRef.current !== owner.projectId) return;
        const current = batchRunRef.current;
        if (!current) return;
        replaceBatchRun(recordBatchEnqueueFailure(current, n, error?.message || (typeof error === "string" ? error : "Could not enqueue image job.")));
        if (stopped() || error?.portableStop) stopEnqueue();
      }
    }
    let prepared = [];
    const submissionLimit = structuredPromptModel ? 1 : BATCH_SUBMISSION_CONCURRENCY;
    while (!stopped() && !owner.enqueueStopped && (prepared.length || !nextChunk.done)) {
      if (owner.submissions.size >= submissionLimit) await Promise.race(owner.submissions);
      await waitForBatchSlot(owner, BATCH_ACTIVE_JOB_LIMIT);
      if (stopped() || owner.enqueueStopped) break;
      if (!prepared.length) prepared = await prepareValidationBatch();
      if (stopped() || owner.enqueueStopped || !prepared.length) break;
      // A response may have opened recovery while we awaited validation or a queue slot.
      if (owner.submissionGroup.recovery) await owner.submissionGroup.recovery.promise;
      if (stopped() || owner.enqueueStopped) break;
      const submission = submitEntry(prepared.shift()).finally(() => {
        owner.submissions.delete(submission);
        releaseBatchSlotWaiters();
      });
      owner.submissions.add(submission);
    }
    // A POST already in flight may commit after Stop. Drain every response so its id
    // is accounted for (or cancelled) before releasing this run's submission ownership.
    await Promise.all(owner.submissions);
    const current = batchRunRef.current;
    if (current && !stopped()) replaceBatchRun({ ...current, submitting: false, expanding: null });
    else if (current && !owner.disposed && batchOwnerRef.current === owner && batchProjectRef.current === owner.projectId) replaceBatchRun({ ...current, submitting: false, expanding: null });
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
      const status = batchItemStatus(jobId, batchJobsRef.current, batchObservedJobsRef.current);
      if (status !== "queued" && status !== "running") {
        continue;
      }
      const job = batchJobsRef.current.find((entry) => entry.id === jobId) ?? batchOwnerRef.current?.jobs.get(jobId);
      if (job) {
        batchJobActionRef.current(job, "cancel");
      }
    }
  }

  function observeJobs(jobs) {
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
  }
  return {
    runBatch, replaceBatchRun, cancelBatchRun, observeJobs,
    progress: (jobs) => batchRunRef.current
      ? summarizeBatchRun(batchRunRef.current, jobs, batchObservedJobsRef.current) : null,
    commit(projectId, jobs, jobAction) {
      batchProjectRef.current = projectId;
      batchJobsRef.current = jobs;
      batchJobActionRef.current = jobAction;
    },
    attachProject() {
      if (batchOwnerRef.current && batchOwnerRef.current.projectId !== batchProjectRef.current) {
        batchOwnerRef.current = null;
        batchObservedJobsRef.current.clear();
        replaceBatchRun(null);
      }
    },
    dispose() {
      const owner = batchOwnerRef.current;
      if (!owner) return;
      owner.stopped = true;
      owner.controller.abort();
      owner.disposed = true;
      releaseBatchSlotWaiters();
      for (const [id, submittedJob] of owner.jobs) {
        const job = batchJobsRef.current.find((entry) => entry.id === id) ?? submittedJob;
        if (job && !terminalStatuses.has(job.status)) batchJobActionRef.current(job, "cancel");
      }
    },
  };
}

// UI admission stays with the page; a running submission retains none of its
// setters, validation drafts, media, or component state.
export function useImageBatchRun(options) {
  const local = useImageBatchController({
    activeProject: options.activeProject, jobs: options.jobs, jobAction: options.jobAction,
    batchRun: options.backgroundImageBatch ? undefined : options.batchRun,
    setBatchRun: options.backgroundImageBatch ? undefined : options.setBatchRun,
  });
  const controller = options.backgroundImageBatch ?? local;
  function runBatch(confirmed = false) {
    if (controller.batchRun?.submitting || !options.activeProject) return;
    const error = options.preflightError("batch");
    if (error) { options.setBatchError(error); return; }
    if (!confirmed && options.batchTotal > options.BATCH_RENDER_CAP) {
      options.setBatchConfirmPending(true);
      return;
    }
    options.setBatchError("");
    options.setBatchConfirmPending(false);
    return controller.runBatch(batchRequestPlan(options));
  }
  return { ...controller, runBatch };
}

function batchRequestPlan({
  BATCH_ACTIVE_JOB_LIMIT, batchPrompts, batchVariables, resolutionOptions, sizeEnvelope,
  structuredPromptModel, batchJobCount, selectedModel, token,
  buildBatchJobRequest, onMagicExpand, batchExpansion, createImageJob,
}) {
  return { BATCH_ACTIVE_JOB_LIMIT, batchPrompts, batchVariables, resolutionOptions, sizeEnvelope,
    structuredPromptModel, batchJobCount, selectedModel, token,
    buildBatchJobRequest, onMagicExpand, batchExpansion, createImageJob };
}
