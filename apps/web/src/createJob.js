import { apiFetch } from "./api.js";
import { handleModelLibraryRejection } from "./modelLibrary.js";
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
    path: "/api/v1/video/jobs",
    buildBody: generationBody,
  },
  audio: {
    path: "/api/v1/audio/jobs",
    buildBody: generationBody,
  },
  image: {
    path: "/api/v1/image/jobs",
    buildBody: generationBody,
  },
  videoUpscale: {
    path: "/api/v1/jobs",
    buildBody: ([payload], project, requestedGpu) => ({
      type: "video_upscale",
      projectId: project.id,
      projectName: project.name,
      requestedGpu,
      payload: { ...payload, projectId: project.id },
    }),
  },
  vqa: {
    path: "/api/v1/image/vqa/jobs",
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
    path: "/api/v1/image/interleave/jobs",
    buildBody: generationBody,
  },
  vector: {
    path: "/api/v1/image/vectorize/jobs",
    buildBody: ([payload], project, requestedGpu) => ({ projectId: project.id, projectName: project.name, requestedGpu, ...payload }),
  },
  vectorPrompt: {
    path: "/api/v1/image/vectorize/prompt/jobs",
    buildBody: generationBody,
  },
});

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
  async function execute(args, options = {}) {
    const { shouldContinue = () => true, detailed = false } = options;
    if (!shouldContinue()) return { cancelled: true };
    if (!project) {
      const message = "Create or open a project first.";
      setError(message);
      return { error: message };
    }
    try {
      if (beforeCreate && (await beforeCreate(...args)) === false) {
        return { error: "Job creation was canceled before submission." };
      }
      // A batch may be stopped while its shared confirmation is open.
      if (!shouldContinue()) return { cancelled: true };
      const job = await apiFetch(definition.path, token, {
        method: "POST",
        body: JSON.stringify(definition.buildBody(args, project, requestedGpu)),
      });
      afterCreate?.(job, ...args);
      setJobs((items) => upsertJobNewest(items, job));
      setError("");
      return { job };
    } catch (err) {
      // A batch accounts for each submission itself. A detached recovery retry could
      // enqueue after it stops and would escape that accounting, so detailed callers
      // receive the reason and may start a new run after resolving the library issue.
      const message = err?.message || (typeof err === "string" ? err : "Could not create job.");
      const prompted = handleModelLibraryRejection(err, detailed ? null : () => createJob(...args));
      if (!prompted) setError(message);
      return detailed ? { error: message } : {};
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
