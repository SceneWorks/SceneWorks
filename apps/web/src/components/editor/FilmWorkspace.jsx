import React, { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import { apiFetch, isAbortError } from "../../api.js";
import {
  applyFilmPlanning,
  cancelFilmPlanning,
  getFilmPlannerAvailability,
  getFilmPlanning,
  installFilmPlanner,
  parseFilmScript,
  preflightFilm,
  startFilmPlanning,
  exportFilmRun,
} from "../../api/films.js";
import { useAppLive, useAppStatic } from "../../context/AppContext.js";
import { errorStatuses, terminalStatuses } from "../../jobTypes.js";
import { FilmReferences } from "./FilmReferences.jsx";
import { FilmBrief } from "./FilmBrief.jsx";
import { FilmLifecycle } from "./FilmLifecycle.jsx";
import { FilmPlanning } from "./FilmPlanning.jsx";
import { FilmRenderOptions } from "./FilmRenderOptions.jsx";
import { FilmReview, normalizedPlan } from "./FilmReview.jsx";
import { FilmShots } from "./FilmShots.jsx";

const FilmSound = lazy(() => import("./FilmSound.jsx").then((module) => ({ default: module.FilmSound })));

const FILM_VIEWS = [
  { id: "brief", label: "Brief" },
  { id: "shots", label: "Shots" },
  { id: "review", label: "Review & Edit" },
];

const QWEN_PLANNER_MODEL_ID = "film_planner_qwen3_6_27b";

export function activeQwenPlannerInstall(jobs = []) {
  return jobs.find((job) =>
    job?.type === "model_download"
    && job?.payload?.modelId === QWEN_PLANNER_MODEL_ID
    && !terminalStatuses.has(job.status)) ?? null;
}

export function latestQwenPlannerInstall(jobs = []) {
  return activeQwenPlannerInstall(jobs)
    ?? jobs.find((job) => job?.type === "model_download" && job?.payload?.modelId === QWEN_PLANNER_MODEL_ID)
    ?? null;
}

export function describeActiveFilmShots(run) {
  const record = run?.record;
  const shots = record?.shots ?? [];
  const active = run?.controllerActive === true && record?.state === "running";
  const selected = new Set([
    ...(run?.locator?.selectedShotIds ?? []),
    ...(record?.selectedShotIds ?? []),
  ]);
  return shots.map((shot) => {
    let status = shot.outcome;
    if (active && status === "not_selected" && selected.has(shot.shotId)) {
      status = shot.attempts?.at(-1)?.status ?? "pending";
    }
    return `${shot.shotId}: ${status}`;
  }).join(" · ");
}

export function FilmWorkspace() {
  const { activeProject, activeTimeline, assets = [], importAsset, models = [], token, refreshTimelines, saveTimeline, setSelectedTimelineId } = useAppStatic();
  const { jobs = [] } = useAppLive();
  const [drafts, setDrafts] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [draft, setDraft] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [plannerAvailability, setPlannerAvailability] = useState(null);
  const [requestedPlannerInstallJob, setRequestedPlannerInstallJob] = useState(null);
  const [plannerInstallError, setPlannerInstallError] = useState("");
  const [planningOperation, setPlanningOperation] = useState(null);
  const [selectedShotIds, setSelectedShotIds] = useState([]);
  const [preflight, setPreflight] = useState(null);
  const [lastRun, setLastRun] = useState(null);
  const [activeView, setActiveView] = useState("brief");
  const viewTabs = useRef([]);
  const selectedDraftId = useRef("");
  const activeTimelineId = useRef("");
  const discoveredTimeline = useRef("");
  const selectedRunExplicit = useRef(false);
  const plannerAvailabilityRequest = useRef(0);
  const activeQueuePlannerInstallJob = activeQwenPlannerInstall(jobs);
  const queuePlannerInstallJob = latestQwenPlannerInstall(jobs);
  const plannerInstallJob = activeQueuePlannerInstallJob?.id !== requestedPlannerInstallJob?.id
    ? (activeQueuePlannerInstallJob ?? requestedPlannerInstallJob ?? queuePlannerInstallJob)
    : (requestedPlannerInstallJob ?? activeQueuePlannerInstallJob ?? queuePlannerInstallJob);
  selectedDraftId.current = draft?.id ?? "";
  activeTimelineId.current = activeTimeline?.id ?? "";

  const acceptRunSnapshot = useCallback((run, { latest = false, select = false } = {}) => {
    const draftId = selectedDraftId.current;
    if (!draftId || run?.locator?.draftId !== draftId) return;
    if (select) selectedRunExplicit.current = true;
    setLastRun((current) => {
      if (select || (latest && !selectedRunExplicit.current) || !current || current.locator?.draftId !== draftId || current.locator?.id === run.locator.id) return run;
      return current;
    });
  }, []);

  useEffect(() => {
    selectedRunExplicit.current = false;
  }, [activeProject?.id, draft?.id]);

  useEffect(() => {
    const projectId = activeProject?.id;
    const draftId = draft?.id;
    const runId = lastRun?.locator?.id;
    const runDraftId = lastRun?.locator?.draftId;
    const timelineId = lastRun?.record?.timeline?.timelineId;
    if (!projectId || !draftId || runDraftId !== draftId || !runId || !timelineId) return undefined;
    const discoveryKey = `${projectId}:${draftId}:${runId}:${timelineId}`;
    if (discoveredTimeline.current === discoveryKey) return undefined;
    let canceled = false;
    let timer = null;
    const controller = new AbortController();
    async function discover(attempt = 0) {
      const result = await refreshTimelines(projectId, { signal: controller.signal });
      if (canceled || selectedDraftId.current !== draftId) return;
      if (result?.ok === false) {
        if (attempt < 2) timer = window.setTimeout(() => discover(attempt + 1), 1000);
        return;
      }
      discoveredTimeline.current = discoveryKey;
      if (!activeTimelineId.current) setSelectedTimelineId(timelineId);
    }
    discover();
    return () => {
      canceled = true;
      controller.abort();
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [activeProject?.id, draft?.id, lastRun?.locator?.draftId, lastRun?.locator?.id, lastRun?.record?.timeline?.timelineId, refreshTimelines, setSelectedTimelineId]);

  useEffect(() => {
    let canceled = false;
    setDrafts([]);
    setDraft(null);
    setSelectedId("");
    setNotice("");
    setPlannerAvailability(null);
    setRequestedPlannerInstallJob(null);
    setPlannerInstallError("");
    setPlanningOperation(null);
    setSelectedShotIds([]);
    setPreflight(null);
    setLastRun(null);
    if (!activeProject?.id) return undefined;
    apiFetch(`/api/v1/projects/${activeProject.id}/films`, token)
      .then((items) => {
        if (canceled) return;
        setDrafts(items);
        if (items[0]) {
          setSelectedId(items[0].id);
          setDraft(items[0]);
          setSelectedShotIds(items[0].productionPlan.shots.map((shot) => shot.id));
        }
      })
      .catch((error) => !canceled && setNotice(error.message));
    return () => { canceled = true; };
  }, [activeProject?.id, token]);

  useEffect(() => {
    const requestId = ++plannerAvailabilityRequest.current;
    setPlannerAvailability(null);
    if (!activeProject?.id || !draft?.id) return undefined;
    let canceled = false;
    getFilmPlannerAvailability(activeProject.id, draft.id, token)
      .then((value) => !canceled && plannerAvailabilityRequest.current === requestId && setPlannerAvailability(value))
      .catch((error) => !canceled && setNotice(error.message));
    getFilmPlanning(activeProject.id, draft.id, token)
      .then((value) => !canceled && setPlanningOperation(value))
      .catch(() => {});
    return () => { canceled = true; };
  }, [activeProject?.id, draft?.id, token]);

  useEffect(() => {
    const jobId = plannerInstallJob?.id;
    if (!jobId || !activeProject?.id || !draft?.id) return undefined;
    let canceled = false;
    let timer = null;
    const controller = new AbortController();

    async function pollInstall() {
      try {
        const job = await apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`, token, { signal: controller.signal });
        if (canceled) return;
        setRequestedPlannerInstallJob(job);
        setPlannerInstallError("");
        if (job.status === "completed") {
          const requestId = ++plannerAvailabilityRequest.current;
          const availability = await getFilmPlannerAvailability(activeProject.id, draft.id, token, { signal: controller.signal });
          if (canceled || plannerAvailabilityRequest.current !== requestId) return;
          setPlannerAvailability(availability);
          const qwen = availability?.providers?.find((item) => item.modelId === "film_planner_qwen3_6_27b");
          setNotice(qwen?.available
            ? "Qwen3.6-27B download completed. Planning did not start automatically."
            : "Qwen3.6-27B download completed, but the planner is still unavailable. Check the model installation before retrying.");
          return;
        }
        if (errorStatuses.has(job.status)) {
          setNotice(job.message || job.error || `Qwen3.6-27B download ${job.status}.`);
          return;
        }
        timer = window.setTimeout(pollInstall, 1000);
      } catch (error) {
        if (canceled || isAbortError(error)) return;
        setPlannerInstallError(error.message);
        setNotice(`Could not refresh the Qwen3.6-27B download: ${error.message}`);
        timer = window.setTimeout(pollInstall, 2000);
      }
    }

    timer = window.setTimeout(pollInstall, 1000);
    return () => {
      canceled = true;
      controller.abort();
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [activeProject?.id, draft?.id, plannerInstallJob?.id, token]);

  async function refreshPlannerAvailability() {
    const projectId = activeProject?.id;
    const draftId = draft?.id;
    if (!projectId || !draftId) return;
    const requestId = ++plannerAvailabilityRequest.current;
    setPlannerAvailability(null);
    try {
      const value = await getFilmPlannerAvailability(projectId, draftId, token);
      if (plannerAvailabilityRequest.current === requestId) {
        setPlannerAvailability(value);
      }
    } catch (error) {
      if (plannerAvailabilityRequest.current === requestId) setNotice(error.message);
    }
  }

  useEffect(() => {
    if (!activeProject?.id || !draft?.id || !["running", "canceling"].includes(planningOperation?.status)) return undefined;
    let canceled = false;
    const timer = window.setTimeout(() => {
      getFilmPlanning(activeProject.id, draft.id, token)
        .then((value) => !canceled && setPlanningOperation(value))
        .catch((error) => !canceled && setNotice(error.message));
    }, 500);
    return () => { canceled = true; window.clearTimeout(timer); };
  }, [activeProject?.id, draft?.id, planningOperation, token]);

  if (!activeProject) return null;

  async function createDraft() {
    setBusy(true);
    setNotice("");
    try {
      const created = await apiFetch(`/api/v1/projects/${activeProject.id}/films`, token, {
        method: "POST",
        body: JSON.stringify({ title: `Film ${drafts.length + 1}` }),
      });
      setDrafts((items) => [created, ...items]);
      setSelectedId(created.id);
      setDraft(created);
      setSelectedShotIds(created.productionPlan.shots.map((shot) => shot.id));
      setPreflight(null);
      setPlanningOperation(null);
      setLastRun(null);
      selectedRunExplicit.current = false;
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function selectDraft(id) {
    setSelectedId(id);
    setNotice("");
    if (!id) {
      setDraft(null);
      setSelectedShotIds([]);
      setPreflight(null);
      return;
    }
    setBusy(true);
    try {
      const loaded = await apiFetch(`/api/v1/projects/${activeProject.id}/films/${id}`, token);
      setDraft(loaded);
      setSelectedShotIds(loaded.productionPlan.shots.map((shot) => shot.id));
      setPreflight(null);
      setPlanningOperation(null);
      setLastRun(null);
      selectedRunExplicit.current = false;
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  function updateDraft(mutator) {
    setPreflight(null);
    setDraft((current) => {
      const next = structuredClone(current);
      mutator(next);
      next.reviewPlan = normalizedPlan(next);
      return next;
    });
  }

  async function saveDraft({ updateLocal = true } = {}) {
    if (!draft) return null;
    const saved = await apiFetch(`/api/v1/projects/${activeProject.id}/films/${draft.id}`, token, {
      method: "PUT",
      body: JSON.stringify({ ...draft, reviewPlan: normalizedPlan(draft) }),
    });
    if (updateLocal) {
      setDraft(saved);
      setDrafts((items) => [saved, ...items.filter((item) => item.id !== saved.id)]);
    }
    return saved;
  }

  function replaceDraft(next) {
    setDraft(next);
    setSelectedShotIds(next.productionPlan.shots.map((shot) => shot.id));
    setPreflight(null);
    setDrafts((items) => [next, ...items.filter((item) => item.id !== next.id)]);
  }

  async function extractBrief() {
    if (!draft) return;
    setBusy(true);
    setNotice("");
    try {
      const structured = await parseFilmScript(activeProject.id, draft.id, draft.originalScript, token);
      updateDraft((next) => { next.structuredBrief = structured; next.brief = structured.synopsis; });
      setNotice("Editable beats and dialogue extracted. Review them before planning.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function generatePlan(maxRepairRounds, llmTimeoutSeconds) {
    setBusy(true);
    setNotice("");
    try {
      const saved = await saveDraft();
      const operation = await startFilmPlanning(activeProject.id, saved.id, maxRepairRounds, llmTimeoutSeconds, token);
      setPlanningOperation(operation);
      setNotice(operation.status === "failed" ? operation.detail : "Planning started. Rendering will not start automatically.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function cancelPlan() {
    try {
      setPlanningOperation(await cancelFilmPlanning(activeProject.id, draft.id, token));
    } catch (error) {
      setNotice(error.message);
    }
  }

  async function applyCandidate() {
    setBusy(true);
    try {
      const saved = await applyFilmPlanning(activeProject.id, draft.id, planningOperation.id, token);
      setDraft(saved);
      setDrafts((items) => [saved, ...items.filter((item) => item.id !== saved.id)]);
      setNotice("Generated candidate replaced the previous edited plan. Review or edit every shot before rendering.");
      setSelectedShotIds(saved.productionPlan.shots.map((shot) => shot.id));
      setPreflight(null);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function installPlanner(modelId) {
    setBusy(true);
    setPlannerInstallError("");
    try {
      const job = await installFilmPlanner(modelId, token);
      if (!job?.id) throw new Error("The Qwen3.6-27B download did not return a queue job.");
      setRequestedPlannerInstallJob(job);
      setNotice(errorStatuses.has(job.status)
        ? (job.message || job.error || `Qwen3.6-27B download ${job.status}.`)
        : `Qwen3.6-27B download ${job.status === "completed" ? "completed" : "queued"}. Planning will not start automatically.`);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function startRun() {
    setBusy(true);
    setNotice("");
    try {
      const saved = await saveDraft();
      const inspected = await preflightFilm(activeProject.id, saved.id, selectedShotIds, token, saved.revision);
      setPreflight(inspected);
      if (!inspected.valid) {
        setNotice("Preflight found fields that must be corrected before rendering.");
        return;
      }
      const created = await apiFetch(`/api/v1/projects/${activeProject.id}/films/${saved.id}/runs`, token, { method: "POST", body: JSON.stringify({ selectedShotIds, expectedDraftRevision: saved.revision }) });
      let run = await apiFetch(`/api/v1/projects/${activeProject.id}/film-runs/${created.locator.id}/start`, token, { method: "POST" });
      selectedRunExplicit.current = true;
      setLastRun(run);
      setNotice(`Rendering ${selectedShotIds.length} selected shot${selectedShotIds.length === 1 ? "" : "s"} in this project.`);
      while (run.controllerActive) {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        run = await apiFetch(`/api/v1/projects/${activeProject.id}/film-runs/${created.locator.id}`, token);
        setLastRun(run);
        const shotProgress = describeActiveFilmShots(run);
        if (shotProgress) setNotice(shotProgress);
      }
      if (!run.record || run.record.state !== "finished") {
        setNotice("The film controller stopped before the shot finished. The run record remains in the project.");
        return;
      }
      const timelineId = run.record?.timeline?.timelineId;
      if (timelineId) {
        setNotice("Shot ready in its film timeline. Export remains a separate editor action.");
      } else {
        const detail = run.record?.diagnostics?.map((item) => item.message).join(" ");
        setNotice(detail || run.record?.stop?.detail || "The run finished without a clip.");
      }
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function exportCurrentCut() {
    const runId = lastRun?.locator?.id;
    const timelineId = lastRun?.record?.timeline?.timelineId;
    if (!runId || !timelineId) return;
    setBusy(true);
    setNotice("");
    try {
      if (activeTimeline?.id === timelineId && saveTimeline) {
        const saved = await saveTimeline(activeTimeline);
        if (!saved) throw new Error("Save the current timeline before exporting.");
      }
      let run = await exportFilmRun(activeProject.id, runId, token);
      setLastRun(run);
      setNotice("Export started from the current saved cut.");
      while (run.controllerActive) {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        run = await apiFetch(`/api/v1/projects/${activeProject.id}/film-runs/${runId}`, token);
        setLastRun(run);
      }
      const exported = run.record?.export;
      if (exported?.status === "completed") {
        setNotice(exported.droppedAudioLayers?.length ? `Export completed with ${exported.droppedAudioLayers.length} dropped audio layer${exported.droppedAudioLayers.length === 1 ? "" : "s"}.` : "Export completed with every placed audio layer.");
      } else {
        setNotice(exported?.error || `Export ${exported?.status ?? "did not complete"}.`);
      }
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function inspectPreflight() {
    if (!draft) return;
    setBusy(true);
    setNotice("");
    try {
      const saved = await saveDraft();
      const inspected = await preflightFilm(activeProject.id, saved.id, selectedShotIds, token, saved.revision);
      setPreflight(inspected);
      setNotice(inspected.valid ? "Preflight passed. Review the effective requests before rendering." : "Preflight found fields that need attention.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  function selectFilmView(view, focus = false) {
    setActiveView(view);
    if (focus) viewTabs.current[FILM_VIEWS.findIndex((item) => item.id === view)]?.focus();
  }

  function handleViewKeyDown(event, index) {
    let next = null;
    if (event.key === "ArrowRight") next = (index + 1) % FILM_VIEWS.length;
    if (event.key === "ArrowLeft") next = (index - 1 + FILM_VIEWS.length) % FILM_VIEWS.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = FILM_VIEWS.length - 1;
    if (next === null) return;
    event.preventDefault();
    selectFilmView(FILM_VIEWS[next].id, true);
  }

  return (
    <details className="ve-film" open>
      <summary>Film workspace</summary>
      <div className="ve-film-toolbar">
        <label>
          Film draft
          <select aria-label="Film draft" disabled={busy} onChange={(event) => selectDraft(event.target.value)} value={selectedId}>
            <option value="">Choose a draft</option>
            {drafts.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}
          </select>
        </label>
        <button disabled={busy} onClick={createDraft} type="button">New film draft</button>
      </div>
      {draft ? (
        <div>
          <div aria-label="Film workspace views" className="ve-film-view-tabs" role="tablist">
            {FILM_VIEWS.map((view, index) => <button
              aria-controls={`film-view-${view.id}`}
              aria-selected={activeView === view.id}
              className={activeView === view.id ? "selected" : ""}
              id={`film-view-tab-${view.id}`}
              key={view.id}
              onClick={() => selectFilmView(view.id)}
              onKeyDown={(event) => handleViewKeyDown(event, index)}
              ref={(node) => { viewTabs.current[index] = node; }}
              role="tab"
              tabIndex={activeView === view.id ? 0 : -1}
              type="button"
            >{view.label}</button>)}
          </div>
          <FilmLifecycle
            draftId={draft.id}
            onRunChange={acceptRunSnapshot}
            projectId={activeProject.id}
            selectedRunId={lastRun?.locator?.id}
            setNotice={setNotice}
            token={token}
          />
          <div aria-labelledby="film-view-tab-brief" hidden={activeView !== "brief"} id="film-view-brief" role="tabpanel">
            <div className="ve-film-form">
              <label>Film title<input value={draft.title} onChange={(event) => updateDraft((next) => { next.title = event.target.value; })} /></label>
            </div>
            <FilmBrief disabled={busy} draft={draft} onChange={updateDraft} onParse={extractBrief} />
            <FilmReferences
              assets={assets}
              busy={busy}
              draft={draft}
              importAsset={importAsset}
              onDraftChange={updateDraft}
              onReplaceDraft={replaceDraft}
              saveDraft={saveDraft}
              setNotice={setNotice}
              token={token}
            />
            <FilmPlanning
              availability={plannerAvailability}
              disabled={busy}
              draft={draft}
              models={models}
              onApply={applyCandidate}
              onCancel={cancelPlan}
              onChange={updateDraft}
              onInstall={installPlanner}
              onRefreshAvailability={refreshPlannerAvailability}
              installError={plannerInstallError}
              installJob={plannerInstallJob}
              onNotice={setNotice}
              onStart={generatePlan}
              operation={planningOperation}
              token={token}
            />
          </div>
          <div aria-labelledby="film-view-tab-shots" hidden={activeView !== "shots"} id="film-view-shots" role="tabpanel">
            <FilmRenderOptions
              disabled={busy}
              draft={draft}
              onChange={updateDraft}
              projectId={activeProject.id}
              token={token}
            />
            <FilmShots
              capabilities={preflight?.capabilities}
              compiled={preflight?.compiled}
              disabled={busy}
              draft={draft}
              findings={preflight?.findings}
              models={models}
              onChange={updateDraft}
              onImportError={setNotice}
              selectedShotIds={selectedShotIds}
              setSelectedShotIds={setSelectedShotIds}
            />
          </div>
          <div aria-labelledby="film-view-tab-review" hidden={activeView !== "review"} id="film-view-review" role="tabpanel">
            <Suspense fallback={<p className="ve-film-help">Loading sound controls…</p>}>
              <FilmSound
                activeProject={activeProject}
                assets={assets}
                disabled={busy}
                draft={draft}
                onChange={updateDraft}
                onReplaceDraft={replaceDraft}
                saveDraft={saveDraft}
                setNotice={setNotice}
                token={token}
              />
            </Suspense>
            <FilmReview
              active={activeView === "review"}
              draft={draft}
              onChange={updateDraft}
              projectId={activeProject.id}
              refreshTimelines={refreshTimelines}
              runControllerActive={Boolean(lastRun?.controllerActive)}
              runLocatorId={lastRun?.locator?.id}
              setNotice={setNotice}
              setSelectedTimelineId={setSelectedTimelineId}
              token={token}
            />
          </div>
          <div className="ve-film-actions">
            <button disabled={busy} onClick={() => saveDraft().then(() => setNotice("Draft saved."), (error) => setNotice(error.message))} type="button">Save draft</button>
            {activeView === "shots" ? <>
              <button disabled={busy || !selectedShotIds.length} onClick={inspectPreflight} type="button">Run preflight</button>
              <button className="ve-generate" disabled={busy || !selectedShotIds.length} onClick={startRun} type="button">Render selected shots</button>
            </> : null}
            {activeView === "review" ? <button disabled={busy || !lastRun?.record?.timeline?.timelineId} onClick={exportCurrentCut} type="button">Export current cut</button> : null}
          </div>
          {lastRun?.record?.export ? <div className="ve-film-export" aria-label="Film export status">
            <strong>Export {lastRun.record.export.status}</strong>
            <span>{lastRun.record.export.stale ? "Older export is stale" : `Timeline revision ${lastRun.record.export.timelineRevision ?? "unknown"}`}</span>
            {lastRun.record.export.assetId ? <span>Asset {lastRun.record.export.assetId}</span> : null}
            {lastRun.record.export.error ? <span role="alert">{lastRun.record.export.error}</span> : null}
            {(lastRun.record.export.droppedAudioLayers ?? []).map((layer, index) => <span key={`${layer.itemId ?? "layer"}:${index}`}>Dropped audio: {layer.displayName ?? layer.itemId ?? JSON.stringify(layer)}</span>)}
          </div> : null}
        </div>
      ) : null}
      {notice ? <p aria-live="polite" className="ve-notice">{notice}</p> : null}
    </details>
  );
}
