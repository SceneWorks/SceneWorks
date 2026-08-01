import { apiFetch } from "./api.js";
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
  return async (...args) => {
    if (!project) {
      setError("Create or open a project first.");
      return null;
    }
    try {
      if (beforeCreate && (await beforeCreate(...args)) === false) {
        return null;
      }
      const job = await apiFetch(definition.path, token, {
        method: "POST",
        body: JSON.stringify(definition.buildBody(args, project, requestedGpu)),
      });
      afterCreate?.(job, ...args);
      setJobs((items) => upsertJobNewest(items, job));
      setError("");
      return job;
    } catch (err) {
      setError(err.message);
      return null;
    }
  };
}
