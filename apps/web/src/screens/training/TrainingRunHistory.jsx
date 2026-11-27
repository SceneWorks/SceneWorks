import React, { useCallback, useEffect, useRef, useState } from "react";
import { trainingApi } from "../../api/trainingResources.js";
import { isAbortError } from "../../api.js";
import { WorkPanel } from "../../components/WorkPanel.jsx";
import { appConfirm } from "../../appConfirm.jsx";

// The parent keys this component by project: an old workspace's response cannot populate a new one.
export function TrainingRunHistory({ projectId, token, refreshKey, renderRuns }) {
  const [page, setPage] = useState({ jobs: [], nextOffset: null, loading: true, error: "" });
  const request = useRef(null);
  const load = useCallback(async (offset = 0, append = false) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setPage((current) => ({ ...current, loading: true, error: "" }));
    try {
      const result = await trainingApi.savedRuns(projectId, token, { offset }, { signal: controller.signal });
      if (controller.signal.aborted) return;
      setPage((current) => ({
        jobs: [...new Map([...(append ? current.jobs : []), ...result.jobs].map((job) => [job.id, job])).values()],
        nextOffset: result.nextOffset,
        loading: false,
        error: "",
      }));
    } catch (error) {
      if (controller.signal.aborted || isAbortError(error)) return;
      setPage((current) => ({ ...current, loading: false, error: error.message || "Could not load saved training runs." }));
    }
  }, [projectId, token]);
  useEffect(() => {
    load();
    return () => request.current?.abort();
  }, [load, refreshKey]);
  const [removing, setRemoving] = useState(() => new Set());
  // Permanent: the run never returns to this list. Its files are kept — a completed run's LoRA
  // stays in the library and a stopped run's checkpoints stay on disk.
  const remove = useCallback(async (job) => {
    const confirmed = await appConfirm({
      title: "Remove saved run?",
      message: "This permanently removes the run from saved training runs and the Queue. It can no longer be continued or resumed from here. Its LoRA, if any, stays in your library.",
      confirmLabel: "Remove run",
      cancelLabel: "Keep run",
      tone: "danger",
    });
    if (!confirmed) return;
    setRemoving((current) => new Set(current).add(job.id));
    try {
      await trainingApi.removeSavedRun(projectId, job.id, token);
      setPage((current) => ({ ...current, jobs: current.jobs.filter((item) => item.id !== job.id), error: "" }));
    } catch (error) {
      setPage((current) => ({ ...current, error: error.message || "Could not remove the saved run." }));
    } finally {
      setRemoving((current) => {
        const next = new Set(current);
        next.delete(job.id);
        return next;
      });
    }
  }, [projectId, token]);

  return (
    <WorkPanel className="training-live-panel" aria-label="Saved training runs"
      eyebrow="Saved training runs"
      hint="Add more steps to a completed LoRA run, or resume a canceled or interrupted run from its last checkpoint. Clearing the Queue keeps these runs available.">
      {page.loading ? <p role="status">Loading saved training runs…</p> : null}
      {page.error ? <p role="alert">{page.error}</p> : null}
      {!page.loading && !page.error && page.jobs.length === 0 ? <p>No completed, canceled or interrupted LoRA runs in this workspace yet.</p> : null}
      {renderRuns(page.jobs, { onRemove: remove, removing })}
      <div className="training-run-actions">
        <button className="secondary-action" type="button" disabled={page.loading} onClick={() => load()}>Refresh runs</button>
        {page.nextOffset != null ? (
          <button className="secondary-action" type="button" disabled={page.loading}
            onClick={() => load(page.nextOffset, true)}>Load more runs</button>
        ) : null}
      </div>
    </WorkPanel>
  );
}
