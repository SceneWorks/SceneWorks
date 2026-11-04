import { useAppDomainValue } from "../context/useAppDomainValue.js";
import { jobActionRequest } from "../api/jobs.js";
import { jobsApi } from "../api/jobsResources.js";
import { useCallback } from "react";
import { upsertJobNewest } from "../sorters.js";

// F-189: extracted from App along its existing concern boundary.
export function useJobActions({
  token, setJobs, setError
}) {
  const jobAction = useCallback(
    async (job, action, options = {}) => {
      try {
        const body =
          action === "duplicate"
            ? {
                ...options.body,
                payloadChanges: {
                  ...options.body?.payloadChanges,
                  duplicatedAt: new Date().toISOString(),
                },
              }
            : (options.body ?? {});
        const updatedJob = await jobActionRequest(job.id, action, token, { method: "POST", body: JSON.stringify(body) });
        setJobs((items) => upsertJobNewest(items, updatedJob));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [setError, setJobs, token],
  );

  // Clear completed items from the queue (issue #1556 / sc-12231). The server
  // soft-hides every terminal job (keeping Generation Stats + generated assets
  // intact) and returns the cleared ids; prune exactly those from the live jobs
  // list so the queue empties immediately without waiting for a refetch. Scoped
  // to the active project filter so it clears exactly what the operator sees
  // ("all" clears every project).
  const clearCompletedJobs = useCallback(
    async (projectId) => {
      try {
        const body = projectId && projectId !== "all" ? { projectId } : {};
        const response = await jobsApi.clear(token, {
          method: "POST",
          body: JSON.stringify(body),
        });
        const clearedIds = new Set(response?.clearedIds ?? []);
        if (clearedIds.size) {
          setJobs((items) => items.filter((job) => !clearedIds.has(job.id)));
        }
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [setError, setJobs, token],
  );

  // Cancel every pending (not-yet-started) item in the queue (sc-13448). The server
  // flips each queued / pending_caption job straight to terminal `canceled` and
  // returns the updated snapshots; upsert them so the cards flip to "Cancelled"
  // immediately without waiting for the SSE round-trip. Scoped to the active project
  // filter so it cancels exactly what the operator sees ("all" cancels every
  // project) — mirroring Clear completed. Active/running jobs are untouched.
  const cancelPendingJobs = useCallback(
    async (projectId) => {
      try {
        const body = projectId && projectId !== "all" ? { projectId } : {};
        const response = await jobsApi.cancelPending(token, {
          method: "POST",
          body: JSON.stringify(body),
        });
        const canceled = response?.jobs ?? [];
        if (canceled.length) {
          setJobs((items) => canceled.reduce((acc, job) => upsertJobNewest(acc, job), items));
        }
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [setError, setJobs, token],
  );

  // Move an explicit Queue-screen selection ahead of all work that was waiting when the action
  // was submitted. The server ignores anything a worker claimed in the meantime, so this updates
  // ordering without becoming a preemption control. Returned snapshots carry their durable
  // queueRank and are upserted immediately; SSE mirrors the same update to other clients.
  const prioritizeJobs = useCallback(
    async (jobIds) => {
      try {
        const response = await jobsApi.prioritize(token, {
          method: "POST",
          body: JSON.stringify({ jobIds }),
        });
        const prioritized = response?.jobs ?? [];
        if (prioritized.length) {
          setJobs((items) => prioritized.reduce((acc, job) => upsertJobNewest(acc, job), items));
        }
        setError("");
        return true;
      } catch (err) {
        setError(err.message);
        return false;
      }
    },
    [setError, setJobs, token],
  );

  // Clear a single completed item from the queue (issue #1556 / sc-12231) — the
  // per-card "×". The server soft-hides just this terminal job; drop it from the
  // live list on success so the card disappears immediately.
  const clearJob = useCallback(
    async (job) => {
      try {
        await jobsApi.clearJob(job.id, token, {
          method: "POST",
          body: JSON.stringify({}),
        });
        setJobs((items) => items.filter((item) => item.id !== job.id));
        setError("");
      } catch (err) {
        setError(err.message);
      }
    },
    [setError, setJobs, token],
  );
  const context = useAppDomainValue({
    jobAction,
    clearCompletedJobs,
    cancelPendingJobs,
    prioritizeJobs,
    clearJob
  });
  return { ...context, context };
}
