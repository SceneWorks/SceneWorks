import React, { useEffect, useState } from "react";
import { apiFetch } from "../../api.js";
import {
  applyFilmPlanning,
  cancelFilmPlanning,
  getFilmPlannerAvailability,
  getFilmPlanning,
  installFilmPlanner,
  parseFilmScript,
  preflightFilm,
  startFilmPlanning,
} from "../../api/films.js";
import { useAppStatic } from "../../context/AppContext.js";
import { FilmReferences } from "./FilmReferences.jsx";
import { FilmBrief } from "./FilmBrief.jsx";
import { FilmLifecycle } from "./FilmLifecycle.jsx";
import { FilmPlanning } from "./FilmPlanning.jsx";
import { FilmShots } from "./FilmShots.jsx";

export function FilmWorkspace() {
  const { activeProject, assets = [], importAsset, models = [], token, refreshTimelines, setSelectedTimelineId } = useAppStatic();
  const [drafts, setDrafts] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [draft, setDraft] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [plannerAvailability, setPlannerAvailability] = useState(null);
  const [planningOperation, setPlanningOperation] = useState(null);
  const [selectedShotIds, setSelectedShotIds] = useState([]);
  const [preflight, setPreflight] = useState(null);

  useEffect(() => {
    let canceled = false;
    setDrafts([]);
    setDraft(null);
    setSelectedId("");
    setNotice("");
    setPlannerAvailability(null);
    setPlanningOperation(null);
    setSelectedShotIds([]);
    setPreflight(null);
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
    if (!activeProject?.id || !draft?.id) return undefined;
    let canceled = false;
    getFilmPlannerAvailability(activeProject.id, draft.id, token)
      .then((value) => !canceled && setPlannerAvailability(value))
      .catch((error) => !canceled && setNotice(error.message));
    getFilmPlanning(activeProject.id, draft.id, token)
      .then((value) => !canceled && setPlanningOperation(value))
      .catch(() => {});
    return () => { canceled = true; };
  }, [activeProject?.id, draft?.id, token]);

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
      return next;
    });
  }

  async function saveDraft() {
    if (!draft) return null;
    const saved = await apiFetch(`/api/v1/projects/${activeProject.id}/films/${draft.id}`, token, {
      method: "PUT",
      body: JSON.stringify(draft),
    });
    setDraft(saved);
    setDrafts((items) => [saved, ...items.filter((item) => item.id !== saved.id)]);
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

  async function generatePlan() {
    setBusy(true);
    setNotice("");
    try {
      const saved = await saveDraft();
      const operation = await startFilmPlanning(activeProject.id, saved.id, token);
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
    try {
      await installFilmPlanner(modelId, token);
      setNotice("Qwen3.6-27B download queued. The built-in planner remains selected until you opt in.");
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
      const inspected = await preflightFilm(activeProject.id, saved.id, selectedShotIds, token);
      setPreflight(inspected);
      if (!inspected.valid) {
        setNotice("Preflight found fields that must be corrected before rendering.");
        return;
      }
      const created = await apiFetch(`/api/v1/projects/${activeProject.id}/films/${saved.id}/runs`, token, { method: "POST", body: JSON.stringify({ selectedShotIds }) });
      let run = await apiFetch(`/api/v1/projects/${activeProject.id}/film-runs/${created.locator.id}/start`, token, { method: "POST" });
      setNotice(`Rendering ${selectedShotIds.length} selected shot${selectedShotIds.length === 1 ? "" : "s"} in this project.`);
      let shownTimelineId = null;
      while (run.controllerActive) {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        run = await apiFetch(`/api/v1/projects/${activeProject.id}/film-runs/${created.locator.id}`, token);
        const readyTimelineId = run.record?.timeline?.timelineId;
        if (readyTimelineId && readyTimelineId !== shownTimelineId) {
          await refreshTimelines(activeProject.id);
          setSelectedTimelineId(readyTimelineId);
          shownTimelineId = readyTimelineId;
        }
        const shots = run.record?.shots ?? [];
        if (shots.length) setNotice(shots.map((shot) => `${shot.shotId}: ${shot.outcome}`).join(" · "));
      }
      if (!run.record || run.record.state !== "finished") {
        setNotice("The film controller stopped before the shot finished. The run record remains in the project.");
        return;
      }
      const timelineId = run.record?.timeline?.timelineId;
      if (timelineId) {
        await refreshTimelines(activeProject.id);
        setSelectedTimelineId(timelineId);
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

  async function inspectPreflight() {
    if (!draft) return;
    setBusy(true);
    setNotice("");
    try {
      const saved = await saveDraft();
      const inspected = await preflightFilm(activeProject.id, saved.id, selectedShotIds, token);
      setPreflight(inspected);
      setNotice(inspected.valid ? "Preflight passed. Review the effective requests before rendering." : "Preflight found fields that need attention.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
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
            onApply={applyCandidate}
            onCancel={cancelPlan}
            onChange={updateDraft}
            onInstall={installPlanner}
            onNotice={setNotice}
            onStart={generatePlan}
            operation={planningOperation}
            token={token}
          />
          <FilmLifecycle
            draftId={draft.id}
            projectId={activeProject.id}
            setNotice={setNotice}
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
          <div className="ve-film-actions">
            <button disabled={busy} onClick={() => saveDraft().then(() => setNotice("Draft saved."), (error) => setNotice(error.message))} type="button">Save draft</button>
            <button disabled={busy || !selectedShotIds.length} onClick={inspectPreflight} type="button">Run preflight</button>
            <button className="ve-generate" disabled={busy || !selectedShotIds.length} onClick={startRun} type="button">Render selected shots</button>
          </div>
        </div>
      ) : null}
      {notice ? <p aria-live="polite" className="ve-notice">{notice}</p> : null}
    </details>
  );
}
