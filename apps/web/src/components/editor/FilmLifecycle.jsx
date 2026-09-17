import React, { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "../../api.js";

function operationLabel(run) {
  const owner = run.controllerOwner || "";
  if (owner.includes("review")) return "Review";
  if (owner.includes("repair") || owner.includes("replace")) return "Repair";
  if (owner.includes("edit")) return "Edit";
  if (owner.includes("decision")) return "Decision";
  return "Render";
}

function runStatus(run) {
  if (run.controllerActive) return "running";
  if (run.controllerInterrupted) return "interrupted";
  if (!run.record) return "ready";
  return run.record.stop?.reason || run.record.outcome || run.record.state;
}

export function FilmLifecycle({ draftId, onRunChange, projectId, selectedRunId = "", token, setNotice }) {
  const [runs, setRuns] = useState([]);
  const [planning, setPlanning] = useState(null);
  const [pending, setPending] = useState("");
  const refreshRequest = useRef(0);

  const refresh = useCallback(async () => {
    if (!projectId || !draftId) return;
    const requestId = ++refreshRequest.current;
    const [runResult, planningResult] = await Promise.allSettled([
      apiFetch(`/api/v1/projects/${projectId}/film-runs`, token),
      apiFetch(`/api/v1/projects/${projectId}/films/${draftId}/planning`, token),
    ]);
    if (refreshRequest.current !== requestId) return;
    if (runResult.status === "fulfilled") {
      const listed = Array.isArray(runResult.value) ? runResult.value : [];
      const scoped = listed.filter((run) => run.locator.draftId === draftId);
      setRuns(scoped);
      for (const [index, run] of scoped.entries()) onRunChange?.(run, { latest: index === 0, select: false });
    } else {
      setNotice(runResult.reason.message);
    }
    if (planningResult.status === "fulfilled") setPlanning(planningResult.value);
  }, [draftId, onRunChange, projectId, setNotice, token]);

  useEffect(() => {
    let canceled = false;
    refresh();
    const interval = window.setInterval(() => {
      if (!canceled) refresh();
    }, 2000);
    return () => {
      canceled = true;
      refreshRequest.current += 1;
      window.clearInterval(interval);
    };
  }, [refresh]);

  async function mutate(runId, action) {
    // A list read started before this mutation cannot overwrite the accepted controller snapshot.
    refreshRequest.current += 1;
    setPending(`${runId}:${action}`);
    try {
      const updated = await apiFetch(`/api/v1/projects/${projectId}/film-runs/${runId}/${action}`, token, { method: "POST" });
      if (updated?.locator?.draftId === draftId) onRunChange?.(updated, { select: true });
      setNotice(action === "cancel" ? "Cancellation requested. Completed takes and spent attempts are preserved." : action === "start" ? "Start requested. Checking saved inputs and worker availability." : "Resume requested. Checking saved attempts and worker availability.");
      await refresh();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setPending("");
    }
  }

  const planningActive = planning && ["running", "canceling"].includes(planning.status);
  if (!planningActive && runs.length === 0) return null;

  return (
    <section aria-label="Film operations" className="ve-film-lifecycle">
      <div className="ve-film-reference-heading">
        <strong>Operations</strong>
        <span>Durable progress remains available after leaving this screen.</span>
      </div>
      {runs.length > 1 ? (
        <label>
          Review run
          <select
            aria-label="Review run"
            onChange={(event) => {
              const selected = runs.find((run) => run.locator.id === event.target.value);
              if (selected) onRunChange?.(selected, { select: true });
            }}
            value={runs.some((run) => run.locator.id === selectedRunId) ? selectedRunId : runs[0].locator.id}
          >
            {runs.map((run) => <option key={run.locator.id} value={run.locator.id}>{run.locator.id} · {runStatus(run)}</option>)}
          </select>
        </label>
      ) : null}
      {planningActive ? (
        <div className="ve-film-lifecycle-row">
          <span><strong>Planning</strong> · {planning.stage}</span>
          <span>{planning.detail || `${Math.round((planning.progress || 0) * 100)}%`}</span>
        </div>
      ) : null}
      {runs.map((run) => {
        const resumable = Boolean(run.record && (run.record.state === "running" || run.record.stop?.resumable));
        const detail = run.record?.stop?.detail;
        return (
          <div className="ve-film-lifecycle-row" key={run.locator.id}>
            <span><strong>{operationLabel(run)}</strong> · {runStatus(run)}</span>
            {detail ? <span>{detail}</span> : null}
            {run.actionOperation?.status === "failed" ? <span role="alert">{run.actionOperation.action} failed: {run.actionOperation.detail}. Correct the problem and retry the action.</span> : null}
            {run.actionOperation?.status === "running" && !run.controllerActive ? <span role="alert">{run.actionOperation.action} was interrupted. Retry to reconcile the saved work.</span> : null}
            <div className="ve-film-operation-actions">
              {run.controllerActive ? <button disabled={Boolean(pending)} onClick={() => mutate(run.locator.id, "cancel")} type="button">Cancel</button> : null}
              {!run.controllerActive && !run.record && run.actionOperation ? <button disabled={Boolean(pending)} onClick={() => mutate(run.locator.id, "start")} type="button">Retry start</button> : null}
              {!run.controllerActive && resumable ? <button disabled={Boolean(pending)} onClick={() => mutate(run.locator.id, "resume")} type="button">Resume</button> : null}
            </div>
          </div>
        );
      })}
    </section>
  );
}
