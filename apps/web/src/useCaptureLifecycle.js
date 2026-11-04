import { useCallback, useEffect, useRef, useState } from "react";
import { makeCreateJob } from "./createJob.js";
import { terminalStatuses } from "./jobTypes.js";

// One owner for staged sources, pending requests, and a bounded live-job watch.
export function useCaptureLifecycle({ token, project, requestedGpu, jobs, setError, onComplete, failureMessage }) {
  const [phase, setPhase] = useState("idle");
  const [jobId, setJobId] = useState(null);
  const [createdJobs, setCreatedJobs] = useState([]);
  const generation = useRef(0);
  const controller = useRef(null);
  const urls = useRef(new Set());
  const callbacks = useRef({ onComplete, failureMessage });
  callbacks.current = { onComplete, failureMessage };
  const ownUrl = useCallback((file) => {
    const url = URL.createObjectURL(file);
    urls.current.add(url);
    return url;
  }, []);
  const releaseUrl = useCallback((url) => {
    if (urls.current.delete(url)) URL.revokeObjectURL(url);
  }, []);
  const invalidate = useCallback(() => {
    generation.current += 1;
    controller.current?.abort();
    for (const url of urls.current) URL.revokeObjectURL(url);
    urls.current.clear();
    setJobId(null);
    setPhase("idle");
  }, []);
  useEffect(() => () => {
    generation.current += 1;
    controller.current?.abort();
    for (const url of urls.current) URL.revokeObjectURL(url);
    urls.current.clear();
  }, []);
  const begin = useCallback(() => {
    controller.current?.abort();
    controller.current = new AbortController();
    return ++generation.current;
  }, []);
  const snapshot = useCallback(() => generation.current, []);
  const current = useCallback((id) => generation.current === id, []);
  const startJob = useCallback(async (id, type, payload) => {
    const create = makeCreateJob({
      definition: { path: "/api/v1/jobs", buildBody: ([value], selectedProject, gpu) => ({
        type, projectId: selectedProject.id, projectName: selectedProject.name ?? null,
        requestedGpu: gpu, payload: { ...value, projectId: selectedProject.id },
      }) }, token, project, requestedGpu, setJobs: (update) => { if (current(id)) setCreatedJobs(update); },
      setError: (message) => { if (current(id)) setError(message); },
    });
    const result = await create.withResult([payload], { shouldContinue: () => current(id), signal: controller.current?.signal });
    if (!current(id)) return { cancelled: true };
    if (result.job) { setJobId(result.job.id); setPhase("detecting"); }
    return result;
  }, [token, project, requestedGpu, setError, current]);
  useEffect(() => {
    if (!jobId) return;
    const job = jobs.find((item) => item.id === jobId) ?? createdJobs.find((item) => item.id === jobId);
    if (!job || !terminalStatuses.has(job.status)) return;
    setJobId(null);
    if (job.status !== "completed") {
      setError(job.error ?? job.message ?? callbacks.current.failureMessage);
      setPhase("idle");
      return;
    }
    callbacks.current.onComplete(job, setPhase);
  }, [jobId, jobs, createdJobs, setError]);
  useEffect(() => {
    if (!jobId) return;
    const timer = setTimeout(() => {
      setJobId(null);
      setPhase("idle");
      setError("Detection timed out while waiting for the job. Check the queue for its status.");
    }, 10 * 60 * 1000);
    return () => clearTimeout(timer);
  }, [jobId, setError]);
  return { phase, setPhase, begin, snapshot, current, startJob, invalidate, ownUrl, releaseUrl };
}
