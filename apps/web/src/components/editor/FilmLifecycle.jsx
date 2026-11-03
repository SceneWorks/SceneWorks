import React, { useCallback, useEffect, useRef, useState } from "react";
import { getFilmPlanning, listFilmRuns, startFilmRun } from "../../api/films.js";

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

export function FilmLifecycle({ draftId, onRunChange, onRuns, planningSnapshot, feedFence, feedPaused = false, feedEpoch = 0, projectId, selectedRunId = "", runActive = false, screenActive = true, token, setNotice }) {
  const [runs, setRuns] = useState([]);
  const [projectRunsActive, setProjectRunsActive] = useState(false);
  const [planning, setPlanning] = useState(null);
  const [pending, setPending] = useState("");
  const refreshRequest = useRef(0);
  const feedController = useRef(null);
  const actionController = useRef(null);
  useEffect(() => {
    setPending("");
    return () => { actionController.current?.abort(); };
  }, [projectId, draftId]);

  const refresh = useCallback(async (ownerSignal) => {
    if (!projectId || !draftId || ownerSignal?.aborted || feedFence?.current.pending) return;
    const generation = feedFence?.current.generation;
    const stale = () => feedFence && (feedFence.current.pending || feedFence.current.generation !== generation);
    feedController.current?.abort();
    const controller = new AbortController();
    feedController.current = controller;
    const abort = () => controller.abort();
    if (feedFence) feedFence.current.abort = abort;
    ownerSignal?.addEventListener("abort", abort, { once: true });
    const signal = controller.signal;
    const requestId = ++refreshRequest.current;
    try {
      const [runResult, planningResult] = await Promise.allSettled([
        listFilmRuns(projectId, token, { signal }),
        getFilmPlanning(projectId, draftId, token, { signal }),
      ]);
      if (signal.aborted || refreshRequest.current !== requestId || stale()) return;
      if (runResult.status === "fulfilled") {
        const listed = Array.isArray(runResult.value) ? runResult.value : [];
        await onRuns?.(listed, signal);
        if (signal.aborted || refreshRequest.current !== requestId || stale()) return;
        setProjectRunsActive(listed.some((run) => run.controllerActive));
        const scoped = listed.filter((run) => run.locator.draftId === draftId);
        setRuns(scoped);
        for (const [index, run] of scoped.entries()) onRunChange?.(run, { latest: index === 0, select: false });
      } else {
        setNotice(runResult.reason.message);
      }
      if (planningResult.status === "fulfilled") setPlanning(planningResult.value);
    } finally {
      if (feedFence?.current.abort === abort) feedFence.current.abort = null;
      ownerSignal?.removeEventListener("abort", abort);
    }
  }, [draftId, feedFence, onRunChange, onRuns, projectId, setNotice, token]);

  const [visible, setVisible] = useState(() => !document.hidden);
  const visibleRef = useRef(screenActive && visible);
  visibleRef.current = screenActive && visible;
  useEffect(() => {
    if (!screenActive || !visible) {
      actionController.current?.abort();
      setPending("");
    }
  }, [screenActive, visible]);
  useEffect(() => {
    const changed = () => setVisible(!document.hidden);
    document.addEventListener("visibilitychange", changed);
    return () => document.removeEventListener("visibilitychange", changed);
  }, []);

  useEffect(() => {
    if (!screenActive || !visible || feedPaused) return undefined;
    const controller = new AbortController();
    refresh(controller.signal);
    return () => { controller.abort(); refreshRequest.current += 1; };
  }, [refresh, screenActive, visible, selectedRunId, feedPaused, feedEpoch]);

  const active = runActive || projectRunsActive;
  useEffect(() => {
    if (!screenActive || !visible || !active || pending || feedPaused) return undefined;
    const controller = new AbortController();
    let timer;
    async function poll() {
      if (visibleRef.current) await refresh(controller.signal);
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 1000);
    }
    timer = window.setTimeout(poll, 1000);
    return () => { controller.abort(); refreshRequest.current += 1; window.clearTimeout(timer); };
  }, [active, pending, refresh, screenActive, visible, feedPaused]);

  async function mutate(runId, action) {
    // A list read started before this mutation cannot overwrite the accepted controller snapshot.
    refreshRequest.current += 1;
    feedController.current?.abort();
    actionController.current?.abort();
    const controller = new AbortController();
    actionController.current = controller;
    setPending(`${runId}:${action}`);
    try {
      const updated = await startFilmRun(projectId, runId, action, token, { signal: controller.signal });
      if (controller.signal.aborted) return;
      if (updated?.locator?.draftId === draftId) onRunChange?.(updated, { select: true });
      setNotice(action === "cancel" ? "Cancellation requested. Completed takes and spent attempts are preserved." : action === "start" ? "Start requested. Checking saved inputs and worker availability." : "Resume requested. Checking saved attempts and worker availability.");
      if (visibleRef.current) await refresh(controller.signal);
    } catch (error) {
      if (!controller.signal.aborted) setNotice(error.message);
    } finally {
      if (!controller.signal.aborted) setPending("");
    }
  }

  const effectivePlanning = planningSnapshot ?? planning;
  const planningActive = effectivePlanning && ["running", "canceling"].includes(effectivePlanning.status);
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
          <span><strong>Planning</strong> · {effectivePlanning.stage}</span>
          <span>{effectivePlanning.detail || `${Math.round((effectivePlanning.progress || 0) * 100)}%`}</span>
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
