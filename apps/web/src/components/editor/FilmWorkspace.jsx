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
import { Icon } from "../Icons.jsx";
import { FilmBrief } from "./FilmBrief.jsx";
import { FilmCutStrip } from "./FilmCutStrip.jsx";
import { FilmLifecycle } from "./FilmLifecycle.jsx";
import { FilmPlanning } from "./FilmPlanning.jsx";
import { FilmRenderOptions } from "./FilmRenderOptions.jsx";
import { FilmReview, normalizedPlan } from "./FilmReview.jsx";
import { FilmShots } from "./FilmShots.jsx";
import { filmRunSummary } from "./filmShotState.js";

const FilmSound = lazy(() => import("./FilmSound.jsx").then((module) => ({ default: module.FilmSound })));

// The film workflow in the order an operator walks it. Every step stays mounted so unsaved
// edits and open disclosures survive moving between steps.
const FILM_VIEWS = [
  { id: "script", label: "Script" },
  { id: "references", label: "References" },
  { id: "plan", label: "Plan" },
  { id: "shots", label: "Shots" },
  { id: "review", label: "Review" },
  { id: "sound", label: "Sound" },
];

function plural(count, noun) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

export function describeFilmSteps({ draft, planningOperation, preflight, selectedShotIds, summary }) {
  const brief = draft.structuredBrief ?? {};
  const beats = brief.beats?.length ?? 0;
  const references = draft.referencePack?.references ?? [];
  const approved = references.filter((entry) => entry.approved).length;
  const findings = preflight?.findings?.length ?? 0;
  const planning = planningOperation?.status;
  return {
    script: { done: beats > 0, status: beats ? `${plural(beats, "beat")} · ${plural(brief.dialogue?.length ?? 0, "line")}` : "Paste a script to begin" },
    references: { done: approved > 0, status: references.length ? `${approved} of ${references.length} approved` : "Optional" },
    plan: { done: planning === "applied", busy: planning === "running" || planning === "canceling", status: planning ? `Planning ${planning}` : "Generate or write shots by hand" },
    shots: { warn: findings > 0, status: `${plural(summary.total, "shot")} · ${selectedShotIds.length} selected${findings ? ` · ${plural(findings, "finding")}` : ""}` },
    review: { warn: summary.decide > 0, status: summary.delivered ? (summary.decide ? `${plural(summary.decide, "take")} to decide` : `${summary.delivered} delivered`) : "Render a shot first" },
    sound: { status: "Dialogue and sound beds" },
  };
}

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

export function FilmWorkspace({ blankTimelineLabel = "New timeline", mode = "film", modeSwitch = null, onEngage, onFilmChange, onNewTimeline, onOpenTimeline, showStart = false, viewRequest = null } = {}) {
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
  const [activeView, setActiveView] = useState("script");
  // A project with no timeline opens on the start paths even when drafts exist; picking or
  // creating a draft (or a hand-off from a timeline clip) is what enters the step view.
  const [engaged, setEngaged] = useState(false);
  const [focusShotId, setFocusShotId] = useState("");
  const explicitRunDraftId = useRef("");
  const viewTabs = useRef([]);
  const selectedDraftId = useRef("");
  const lastRunId = useRef("");
  const selectDraftRef = useRef(null);
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
  lastRunId.current = lastRun?.locator?.id ?? "";

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
    selectedRunExplicit.current = Boolean(draft?.id) && explicitRunDraftId.current === draft.id;
    explicitRunDraftId.current = "";
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
    setEngaged(false);
    setFocusShotId("");
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

  useEffect(() => {
    onFilmChange?.({ draft, run: lastRun });
  }, [draft, lastRun, onFilmChange]);

  // A timeline clip hands off with the run and shot it was delivered by, so Review opens on
  // that film even when another draft or an older run is the one currently selected.
  useEffect(() => {
    if (!viewRequest?.view) return undefined;
    setActiveView(viewRequest.view);
    setEngaged(true);
    setFocusShotId(viewRequest.shotId ?? "");
    const projectId = activeProject?.id;
    const runId = viewRequest.runId;
    if (!projectId || !runId || lastRunId.current === runId) return undefined;
    let canceled = false;
    (async () => {
      try {
        const run = await apiFetch(`/api/v1/projects/${projectId}/film-runs/${encodeURIComponent(runId)}`, token);
        const draftId = run?.locator?.draftId;
        if (canceled || !draftId) return;
        if (selectedDraftId.current !== draftId) {
          explicitRunDraftId.current = draftId;
          await selectDraftRef.current(draftId);
          if (canceled) return;
        }
        selectedRunExplicit.current = true;
        setLastRun(run);
      } catch (error) {
        if (!canceled) setNotice(`Could not open the film run for this clip: ${error.message}`);
      }
    })();
    return () => { canceled = true; };
  }, [activeProject?.id, token, viewRequest]);

  if (!activeProject) return null;

  function engage() {
    setEngaged(true);
    onEngage?.();
  }

  async function createDraft() {
    engage();
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
    if (id) engage();
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

  selectDraftRef.current = selectDraft;

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
    engage();
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
      setNotice(`Render requested for ${selectedShotIds.length} selected shot${selectedShotIds.length === 1 ? "" : "s"}. Checking inputs and worker availability.`);
      while (run.controllerActive) {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        run = await apiFetch(`/api/v1/projects/${activeProject.id}/film-runs/${created.locator.id}`, token);
        setLastRun(run);
        const shotProgress = describeActiveFilmShots(run);
        if (shotProgress) setNotice(shotProgress);
      }
      if (run.actionOperation?.status === "failed") {
        setNotice(`${run.actionOperation.action} failed: ${run.actionOperation.detail}. Correct the problem and retry the action.`);
        return;
      }
      if (!run.record || run.record.state !== "finished") {
        setNotice("The film controller stopped before the shot finished. Saved progress remains in the project.");
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
    if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % FILM_VIEWS.length;
    if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + FILM_VIEWS.length) % FILM_VIEWS.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = FILM_VIEWS.length - 1;
    if (next === null) return;
    event.preventDefault();
    selectFilmView(FILM_VIEWS[next].id, true);
  }

  const summary = filmRunSummary(draft, lastRun);
  const steps = draft ? describeFilmSteps({ draft, planningOperation, preflight, selectedShotIds, summary }) : {};
  const filmTimelineId = lastRun?.record?.timeline?.timelineId;
  const exported = lastRun?.record?.export;

  function selectCutShot(shotId, state) {
    selectFilmView(state === "planned" || state === "queued" || state === "failed" ? "shots" : "review");
  }

  return (
    <section aria-label="Film workspace" className="ve-film" hidden={mode !== "film"}>
      <div className="ve-toolbar ve-film-bar">
        <div className="ve-brand">
          <span className="ve-mark"><Icon.Video size={14} /></span>
          <div className="ve-brand-text">
            <strong>{draft?.title || "Film"}</strong>
            <span className="ve-brand-sub">{draft ? `film draft · revision ${draft.revision ?? 1}` : "script to timeline"}</span>
          </div>
        </div>
        {modeSwitch ? <><div className="ve-toolbar-div" />{modeSwitch}</> : null}
        <div className="ve-toolbar-div" />
        <select aria-label="Film draft" className="ve-select ve-film-draft-select" disabled={busy} onChange={(event) => selectDraft(event.target.value)} value={selectedId}>
          <option value="">Choose a draft</option>
          {drafts.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}
        </select>
        <button aria-label="New film draft" className="ve-ghost-btn ve-film-bar-text" disabled={busy} onClick={createDraft} title="New film draft" type="button"><Icon.Plus size={15} /><span>New film draft</span></button>
        <div className="ve-toolbar-right">
          {summary.renderingShotId || (lastRun?.controllerActive && summary.total) ? <span className="ve-film-run-chip">{summary.renderingShotId ? `${summary.renderingShotId} rendering` : "Run active"} · {summary.delivered} of {summary.total}</span> : null}
          {draft ? <button aria-label="Save draft" className="ve-ghost-btn ve-film-bar-text" disabled={busy} title="Save draft" onClick={() => saveDraft().then(() => setNotice("Draft saved."), (error) => setNotice(error.message))} type="button"><Icon.Save size={15} /><span>Save draft</span></button> : null}
          {draft ? <button className="ve-export" disabled={busy || !filmTimelineId} onClick={exportCurrentCut} title={filmTimelineId ? "" : "Render a shot to create the film timeline."} type="button">Export current cut</button> : null}
        </div>
      </div>
      {draft && (engaged || !showStart) ? (
        <div className="ve-film-body">
          <aside className="ve-film-steps">
            <span className="ve-film-eyebrow">Film workflow</span>
            <div aria-label="Film workspace views" aria-orientation="vertical" className="ve-film-view-tabs" role="tablist">
              {FILM_VIEWS.map((view, index) => {
                const step = steps[view.id] ?? {};
                const tone = step.busy ? "busy" : step.warn ? "warn" : step.done ? "done" : "";
                return (
                  <div className={`ve-film-step ${tone} ${activeView === view.id ? "selected" : ""}`} key={view.id} role="presentation">
                    <button
                      aria-controls={`film-view-${view.id}`}
                      aria-describedby={`film-view-status-${view.id}`}
                      aria-selected={activeView === view.id}
                      id={`film-view-tab-${view.id}`}
                      onClick={() => selectFilmView(view.id)}
                      onKeyDown={(event) => handleViewKeyDown(event, index)}
                      ref={(node) => { viewTabs.current[index] = node; }}
                      role="tab"
                      tabIndex={activeView === view.id ? 0 : -1}
                      type="button"
                    >{view.label}</button>
                    <span className="ve-film-step-status" id={`film-view-status-${view.id}`}>{step.status}</span>
                  </div>
                );
              })}
            </div>
            <FilmLifecycle
              draftId={draft.id}
              onRunChange={acceptRunSnapshot}
              projectId={activeProject.id}
              selectedRunId={lastRun?.locator?.id}
              setNotice={setNotice}
              token={token}
            />
          </aside>
          <div className="ve-film-main">
            <div className="ve-film-panels">
              <div aria-labelledby="film-view-tab-script" hidden={activeView !== "script"} id="film-view-script" role="tabpanel">
                <div className="ve-film-form">
                  <label>Film title<input value={draft.title} onChange={(event) => updateDraft((next) => { next.title = event.target.value; })} /></label>
                </div>
                <FilmBrief disabled={busy} draft={draft} onChange={updateDraft} onParse={extractBrief} />
              </div>
              <div aria-labelledby="film-view-tab-references" hidden={activeView !== "references"} id="film-view-references" role="tabpanel">
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
              </div>
              <div aria-labelledby="film-view-tab-plan" hidden={activeView !== "plan"} id="film-view-plan" role="tabpanel">
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
                  run={lastRun}
                  selectedShotIds={selectedShotIds}
                  setSelectedShotIds={setSelectedShotIds}
                />
              </div>
              <div aria-labelledby="film-view-tab-review" hidden={activeView !== "review"} id="film-view-review" role="tabpanel">
                <FilmReview
                  active={mode === "film" && activeView === "review"}
                  draft={draft}
                  focusShotId={focusShotId}
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
              <div aria-labelledby="film-view-tab-sound" hidden={activeView !== "sound"} id="film-view-sound" role="tabpanel">
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
              </div>
            </div>
            {exported ? <div className="ve-film-export" aria-label="Film export status">
              <strong>Export {exported.status}</strong>
              <span>{exported.stale ? "Older export is stale" : `Timeline revision ${exported.timelineRevision ?? "unknown"}`}</span>
              {exported.assetId ? <span>Asset {exported.assetId}</span> : null}
              {exported.error ? <span role="alert">{exported.error}</span> : null}
              {(exported.droppedAudioLayers ?? []).map((layer, index) => <span key={`${layer.itemId ?? "layer"}:${index}`}>Dropped audio: {layer.displayName ?? layer.itemId ?? JSON.stringify(layer)}</span>)}
            </div> : null}
            {notice ? <p aria-live="polite" className="ve-notice">{notice}</p> : null}
            {activeView === "shots" ? <div className="ve-film-actions ve-film-footer">
              <span>{plural(selectedShotIds.length, "shot")} selected</span>
              <button disabled={busy || !selectedShotIds.length} onClick={inspectPreflight} type="button">Run preflight</button>
              <button className="ve-generate" disabled={busy || !selectedShotIds.length} onClick={startRun} type="button">Render selected shots</button>
            </div> : null}
            <FilmCutStrip draft={draft} onOpenTimeline={onOpenTimeline} onSelectShot={selectCutShot} run={lastRun} />
          </div>
        </div>
      ) : (
        <div className="ve-film-start">
          <div className="ve-film-start-head">
            <h2>How do you want to start?</h2>
            <p>Both paths end in the same editable timeline. You can switch between Film and Timeline at any time.</p>
          </div>
          <div className="ve-film-start-cards">
            <article className="work-panel ve-film-start-card">
              <div className="work-panel-rule" />
              <span className="ve-film-start-icon primary"><Icon.Video size={20} /></span>
              <h3>Start from a script</h3>
              <p>Turn prose or a screenplay into a shot plan, then render consistent clips in sequence.</p>
              <ul>
                <li>Extract beats and dialogue</li>
                <li>Bind character and location references</li>
                <li>Render shots straight into a timeline</li>
              </ul>
              <button className="primary-action" disabled={busy} onClick={createDraft} type="button">New film draft</button>
            </article>
            <article className="ve-film-start-card">
              <span className="ve-film-start-icon"><Icon.Editor size={20} /></span>
              <h3>Start with a blank timeline</h3>
              <p>Assemble clips you already have, and generate extensions or bridges as you cut.</p>
              <ul>
                <li>Add assets from the media bin</li>
                <li>Extend, regenerate, or bridge gaps</li>
                <li>Add audio tracks</li>
              </ul>
              <button className="secondary-action" onClick={onNewTimeline} type="button">{blankTimelineLabel}</button>
            </article>
          </div>
          {drafts.length ? <div className="ve-film-start-drafts">
            <span className="ve-film-eyebrow">Continue a film draft</span>
            {drafts.map((item) => <button disabled={busy} key={item.id} onClick={() => selectDraft(item.id)} type="button"><Icon.Video size={15} /><strong>{item.title}</strong><span>{plural(item.productionPlan?.shots?.length ?? 0, "shot")}</span><Icon.ArrowRight size={14} /></button>)}
          </div> : null}
          {notice ? <p aria-live="polite" className="ve-notice">{notice}</p> : null}
        </div>
      )}
    </section>
  );
}
