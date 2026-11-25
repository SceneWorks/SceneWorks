import { useEffect, useInsertionEffect, useRef, useState } from "react";
import { useAppDomainValue } from "../context/useAppDomainValue.js";

// The shell owns submission lifetime, but loads the batch machinery only when
// a run starts. Inactive studio code and request builders stay out of startup.
export function useImageBatchController({
  activeProject, jobs, jobAction, batchRun: providedRun, setBatchRun: providedSetRun,
}) {
  const [ownedRun, setOwnedRun] = useState(null);
  const batchRun = providedSetRun ? providedRun : ownedRun;
  const hostRef = useRef(null);
  if (!hostRef.current) hostRef.current = createBatchHost(providedSetRun ?? setOwnedRun);
  const host = hostRef.current;
  useInsertionEffect(() => {
    host.commit(activeProject?.id, jobs, jobAction);
  }, [host, activeProject?.id, jobs, jobAction]);
  useEffect(() => {
    host.attachProject();
    return () => host.dispose();
  }, [host, activeProject?.id]);
  useEffect(() => { host.observeJobs(jobs); }, [host, batchRun, jobs]);
  return useAppDomainValue({ batchRun,
    runBatch: host.runBatch, replaceBatchRun: host.replaceBatchRun,
    cancelBatchRun: host.cancelBatchRun, batchRunProgress: host.progress(jobs),
  });
}

// This owner receives only a publisher. An async start cannot retain the
// catalogs or media from the React render in which the user clicked Run.
function createBatchHost(publish) {
  let environment = { projectId: null, jobs: [], jobAction: null };
  let runtime = null;
  let epoch = 0;
  let starting = false;
  let loadFailure = null;
  return {
    async runBatch(plan) {
      if (starting || !environment.projectId) return;
      starting = true;
      const started = epoch;
      const projectId = environment.projectId;
      try {
        let createImageBatchRuntime;
        try {
          ({ createImageBatchRuntime } = await import("../screens/imageStudio/useImageBatchRun.js"));
        } catch (error) {
          if (started !== epoch || environment.projectId !== projectId) return;
          const total = plan.batchJobCount;
          const message = `Could not load batch processing: ${error?.message || "Please try again."}`;
          loadFailure = { projectId, progress: { total, pending: total, queued: 0, running: 0,
            completed: 0, failed: 0, done: 0, active: 0, allDone: false } };
          publish({ total, submitted: 0, completed: 0, failed: 0, activeJobIds: [],
            submitting: false, firstError: message, lastError: message });
          return;
        }
        if (started !== epoch || environment.projectId !== projectId) return;
        loadFailure = null;
        runtime ??= createImageBatchRuntime(publish);
        runtime.commit(environment.projectId, environment.jobs, environment.jobAction);
        runtime.attachProject();
        return await runtime.runBatch(plan);
      } finally {
        if (started === epoch) starting = false;
      }
    },
    replaceBatchRun(next) { loadFailure = null; if (runtime) runtime.replaceBatchRun(next); else publish(next); },
    cancelBatchRun() { epoch += 1; starting = false; runtime?.cancelBatchRun(); },
    commit(projectId, jobs, jobAction) {
      environment = { projectId, jobs, jobAction };
      runtime?.commit(projectId, jobs, jobAction);
    },
    attachProject() {
      if (loadFailure && loadFailure.projectId !== environment.projectId) { loadFailure = null; publish(null); }
      runtime?.attachProject();
    },
    observeJobs(jobs) { runtime?.observeJobs(jobs); },
    progress(jobs) { return runtime?.progress(jobs) ?? loadFailure?.progress ?? null; },
    dispose() { epoch += 1; starting = false; runtime?.dispose(); },
  };
}
