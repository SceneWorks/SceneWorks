import { JOB_PATHS, submitJobRequest } from "./api/jobs.js";

import { handleModelLibraryRejection, modelLibraryContext } from "./modelLibrary.js";
import { upsertJobNewest } from "./sorters.js";

function generationBody([payload], project, requestedGpu) {
  return {
    ...payload,
    projectId: project.id,
    projectName: project.name,
    requestedGpu,
  };
}

export const CREATE_JOB_DEFINITIONS = Object.freeze({
  video: {
    path: JOB_PATHS.video,
    buildBody: generationBody,
  },
  audio: {
    path: JOB_PATHS.audio,
    buildBody: generationBody,
  },
  image: {
    path: JOB_PATHS.image,
    buildBody: generationBody,
  },
  videoUpscale: {
    path: JOB_PATHS.generic,
    buildBody: ([payload], project, requestedGpu) => ({
      type: "video_upscale",
      projectId: project.id,
      projectName: project.name,
      requestedGpu,
      payload: { ...payload, projectId: project.id },
    }),
  },
  vqa: {
    path: JOB_PATHS.vqa,
    buildBody: ([asset, question, maxNewTokens], project, requestedGpu) => ({
      projectId: project.id,
      projectName: project.name,
      sourceAssetId: asset.id,
      question,
      maxNewTokens,
      requestedGpu,
    }),
  },
  interleave: {
    path: JOB_PATHS.interleave,
    buildBody: generationBody,
  },
  vector: {
    path: JOB_PATHS.vector,
    buildBody: ([payload], project, requestedGpu) => ({ projectId: project.id, projectName: project.name, requestedGpu, ...payload }),
  },
  vectorPrompt: {
    path: JOB_PATHS.vectorPrompt,
    buildBody: generationBody,
  },
});

// One batch owns one confirmation decision and one model-library recovery at a time.
// Keep these scoped to the run; unrelated creators retain their independent lifecycle.
export function createJobSubmissionGroup() {
  return { confirmation: null, recovery: null };
}

// Canonical project-scoped POST-job wrapper. Each definition owns only its route
// and body shape; project gating, error handling, and newest-first job insertion
// stay identical for every creator.
export function makeCreateJob({
  definition,
  token,
  project,
  requestedGpu,
  setJobs,
  setError,
  beforeCreate,
  afterCreate,
}) {
  async function execute(args, options = {}, operation = Symbol("submission")) {
    const { shouldContinue = () => true, detailed = false, signal, submissionGroup } = options;
    if (!shouldContinue()) return { cancelled: true };
    if (!project) {
      const message = "Create or open a project first.";
      setError(message);
      return { error: message };
    }
    try {
      if (beforeCreate) {
        let abort;
        const confirmation = submissionGroup?.confirmation ?? Promise.resolve(beforeCreate(...args));
        if (submissionGroup) submissionGroup.confirmation = confirmation;
        const cancelled = Symbol("cancelled");
        const decision = signal
          ? await Promise.race([confirmation, new Promise((resolve) => {
              abort = () => resolve(cancelled);
              if (signal.aborted) abort();
              else signal.addEventListener("abort", abort, { once: true });
            })]).finally(() => signal.removeEventListener("abort", abort))
          : await confirmation;
        if (decision === cancelled) return { cancelled: true };
        if (decision === false) return submissionGroup
          ? { cancelled: true }
          : { error: "Job creation was canceled before submission." };
      }
      // A batch may be stopped while its shared confirmation is open.
      if (!shouldContinue()) return { cancelled: true };
      const body = definition.buildBody(args, project, requestedGpu);
      if (!shouldContinue()) return { cancelled: true };
      const job = await submitJobRequest(definition.path, token, {
        method: "POST",
        body: JSON.stringify(body),
      });
      afterCreate?.(job, ...args);
      setJobs((items) => upsertJobNewest(items, job));
      setError("");
      return { job };
    } catch (err) {
      const message = err?.message || (typeof err === "string" ? err : "Could not create job.");
      if (detailed) {
        const sharedRecovery = submissionGroup?.recovery;
        if (sharedRecovery && sharedRecovery.owner !== operation && modelLibraryContext(err)) {
          const result = await sharedRecovery.promise;
          if (result.cancelled || result.stop) return { ...result, job: undefined };
          return execute(args, options, operation);
        }
        // Keep the failed chunk owned by its batch while the recovery dialog is open.
        // Cancel, relocation, a competing dialog, and batch teardown release the wait.
        let resolveRecovery;
        const recovery = new Promise((resolve) => { resolveRecovery = resolve; });
        let waiting = true;
        const finish = (result) => {
          if (!waiting) return;
          waiting = false;
          signal?.removeEventListener("abort", abort);
          resolveRecovery(result);
        };
        const abort = () => finish({ cancelled: true });
        const resume = async () => {
          if (!waiting) return null;
          waiting = false;
          signal?.removeEventListener("abort", abort);
          const result = await execute(args, options, operation);
          resolveRecovery(result);
          return result.job ?? null;
        };
        resume.onAbandon = () => finish({ error: message, stop: true });
        if (handleModelLibraryRejection(err, resume)) {
          const shared = { owner: operation, promise: recovery };
          if (submissionGroup) submissionGroup.recovery = shared;
          if (signal?.aborted || !shouldContinue()) abort();
          else if (waiting) signal?.addEventListener("abort", abort, { once: true });
          return recovery.finally(() => {
            if (submissionGroup?.recovery === shared) submissionGroup.recovery = null;
          });
        }
      } else if (handleModelLibraryRejection(err, () => createJob(...args))) {
        return {};
      }
      setError(message);
      return { error: message };
    }
  }

  async function createJob(...args) {
    const result = await execute(args);
    return result.job ?? null;
  }
  // Opt-in result and cancellation contract for callers that own a multi-job run.
  // Ordinary callers retain the historical job-or-null return and recovery behavior.
  createJob.withResult = (args, options = {}) => execute(args, { ...options, detailed: true });
  return createJob;
}
