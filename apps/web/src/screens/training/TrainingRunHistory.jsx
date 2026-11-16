import React, { useCallback, useEffect, useRef, useState } from "react";
import { trainingApi } from "../../api/trainingResources.js";
import { isAbortError } from "../../api.js";
import { WorkPanel } from "../../components/WorkPanel.jsx";

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
      const result = await trainingApi.completedRuns(projectId, token, { offset }, { signal: controller.signal });
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

  return (
    <WorkPanel className="training-live-panel" aria-label="Completed training runs"
      eyebrow="Completed training runs"
      hint="Select a saved LoRA run to add more steps. Clearing the Queue keeps these runs available.">
      {page.loading ? <p role="status">Loading saved training runs…</p> : null}
      {page.error ? <p role="alert">{page.error}</p> : null}
      {!page.loading && !page.error && page.jobs.length === 0 ? <p>No completed LoRA runs in this workspace yet.</p> : null}
      {renderRuns(page.jobs)}
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
