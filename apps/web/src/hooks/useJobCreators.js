import { useAppDomainValue } from "../context/useAppDomainValue.js";
import { jobsApi } from "../api/jobsResources.js";
import { useCallback, useMemo } from "react";
import { CREATE_JOB_DEFINITIONS, makeCreateJob } from "../createJob.js";

// F-189: extracted from App along its existing concern boundary.
export function useJobCreators({
  token, activeProject, requestedGpu, setJobs, setError, setActiveView, activeViewRef, confirmWorkflowEmbeddingBeforeGeneration
}) {
  // sc-4194: defined here (above the data hooks) because useTimelines takes it as a
  // dependency; a stable identity keeps the timeline hook's queue action stable too.
  const createVideoJob = useMemo(
    () =>
      makeCreateJob({
        definition: CREATE_JOB_DEFINITIONS.video,
        token,
        project: activeProject,
        requestedGpu,
        setJobs,
        setError,
        afterCreate: (_job, _payload, { navigateToQueue = false } = {}) => {
          if (navigateToQueue) {
            setActiveView("Queue");
          }
        },
      }),
    [token, activeProject, requestedGpu, setJobs, setError, setActiveView],
  );

  // SceneWorks Audio Studio (epic 13400 / sc-13404): the audio analogue of createVideoJob. POSTs to
  // the typed /api/v1/audio/jobs endpoint, which resolves + injects the model's manifest entry and
  // enqueues an `audio_generate` job the worker routes to the candle audio lane. `payload` carries
  // { modelId → model, prompt, voice, language, targetDurationSecs, seed }.
  const createAudioJob = useMemo(
    () =>
      makeCreateJob({
        definition: CREATE_JOB_DEFINITIONS.audio,
        token,
        project: activeProject,
        requestedGpu,
        setJobs,
        setError,
      }),
    [token, activeProject, requestedGpu, setJobs, setError],
  );

  const createPlaceholderJob = useCallback(
    async (event, prompt) => {
      event.preventDefault();
      try {
        await jobsApi.list(token, {
          method: "POST",
          body: JSON.stringify({
            type: "placeholder",
            projectId: activeProject?.id ?? null,
            projectName: activeProject?.name ?? null,
            requestedGpu,
            payload: {
              prompt,
              createdFrom: activeViewRef.current,
            },
          }),
        });
        setActiveView("Queue");
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [token, activeProject?.id, activeProject?.name, requestedGpu, activeViewRef, setActiveView, setError],
  );

  const createImageJob = useMemo(
    () =>
      makeCreateJob({
        definition: CREATE_JOB_DEFINITIONS.image,
        token,
        project: activeProject,
        requestedGpu,
        setJobs,
        setError,
        // Every image-producing studio shares this preflight. The first undecided request waits
        // here, before the POST, until the recipe-sharing choice is durably saved.
        beforeCreate: confirmWorkflowEmbeddingBeforeGeneration,
      }),
    [token, activeProject, requestedGpu, setJobs, setError, confirmWorkflowEmbeddingBeforeGeneration],
  );

  const createVectorJob = useMemo(
    () => makeCreateJob({ definition: CREATE_JOB_DEFINITIONS.vector, token, project: activeProject, requestedGpu, setJobs, setError }),
    [token, activeProject, requestedGpu, setJobs, setError],
  );

  const createVectorPromptWorkflow = useMemo(
    () => makeCreateJob({ definition: CREATE_JOB_DEFINITIONS.vectorPrompt, token, project: activeProject, requestedGpu, setJobs, setError }),
    [token, activeProject, requestedGpu, setJobs, setError],
  );

  // Standalone video upscale (epic 4811 / sc-4816): the net-new `video_upscale` job runs
  // on the generic /api/v1/jobs endpoint (like image_upscale), not the generation video
  // endpoint. `payload` carries { sourceAssetId, factor, engine, softness, model, displayName }.
  const createVideoUpscaleJob = useMemo(
    () =>
      makeCreateJob({
        definition: CREATE_JOB_DEFINITIONS.videoUpscale,
        token,
        project: activeProject,
        requestedGpu,
        setJobs,
        setError,
      }),
    [token, activeProject, requestedGpu, setJobs, setError],
  );

  const createVqaJob = useMemo(
    () =>
      makeCreateJob({
        definition: CREATE_JOB_DEFINITIONS.vqa,
        token,
        project: activeProject,
        requestedGpu,
        setJobs,
        setError,
      }),
    [token, activeProject, requestedGpu, setJobs, setError],
  );

  const createInterleaveJob = useMemo(
    () =>
      makeCreateJob({
        definition: CREATE_JOB_DEFINITIONS.interleave,
        token,
        project: activeProject,
        requestedGpu,
        setJobs,
        setError,
      }),
    [token, activeProject, requestedGpu, setJobs, setError],
  );
  const context = useAppDomainValue({
    createVideoJob,
    createAudioJob,
    createPlaceholderJob,
    createImageJob,
    createVectorJob,
    createVectorPromptWorkflow,
    createVideoUpscaleJob,
    createVqaJob,
    createInterleaveJob
  });
  return { ...context, context };
}
