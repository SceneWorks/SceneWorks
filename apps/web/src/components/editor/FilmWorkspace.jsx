import React, { useEffect, useState } from "react";
import { apiFetch } from "../../api.js";
import {
  applyFilmPlanning,
  cancelFilmPlanning,
  getFilmPlannerAvailability,
  getFilmPlanning,
  installFilmPlanner,
  parseFilmScript,
  startFilmPlanning,
} from "../../api/films.js";
import { useAppStatic } from "../../context/AppContext.js";
import { FilmReferences } from "./FilmReferences.jsx";
import { FilmBrief } from "./FilmBrief.jsx";
import { FilmLifecycle } from "./FilmLifecycle.jsx";
import { FilmPlanning } from "./FilmPlanning.jsx";

export function FilmWorkspace() {
  const { activeProject, assets, importAsset, token, refreshTimelines, setSelectedTimelineId } = useAppStatic();
  const [drafts, setDrafts] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [draft, setDraft] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [plannerAvailability, setPlannerAvailability] = useState(null);
  const [planningOperation, setPlanningOperation] = useState(null);

  useEffect(() => {
    let canceled = false;
    setDrafts([]);
    setDraft(null);
    setSelectedId("");
    setNotice("");
    setPlannerAvailability(null);
    setPlanningOperation(null);
    if (!activeProject?.id) return undefined;
    apiFetch(`/api/v1/projects/${activeProject.id}/films`, token)
      .then((items) => {
        if (canceled) return;
        setDrafts(items);
        if (items[0]) {
          setSelectedId(items[0].id);
          setDraft(items[0]);
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
      return;
    }
    setBusy(true);
    try {
      setDraft(await apiFetch(`/api/v1/projects/${activeProject.id}/films/${id}`, token));
      setPlanningOperation(null);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  function updateDraft(mutator) {
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
      const created = await apiFetch(`/api/v1/projects/${activeProject.id}/films/${saved.id}/runs`, token, { method: "POST" });
      let run = await apiFetch(`/api/v1/projects/${activeProject.id}/film-runs/${created.locator.id}/start`, token, { method: "POST" });
      setNotice("Rendering the shot in this project.");
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

  const shot = draft?.productionPlan?.shots?.[0];
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
      {draft && shot ? (
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
            onStart={generatePlan}
            operation={planningOperation}
          />
          <FilmLifecycle
            draftId={draft.id}
            projectId={activeProject.id}
            setNotice={setNotice}
            token={token}
          />
          <section aria-labelledby="film-manual-shot-heading" className="ve-film-section">
            <h3 id="film-manual-shot-heading">Current shot plan</h3>
            {draft.productionPlan.shots.map((currentShot, index) => (
              <article className="ve-film-shot-editor" key={currentShot.id}>
                <h4>{currentShot.id}</h4>
                <div className="ve-film-form">
                  <label className="ve-film-prompt">Shot prompt<textarea aria-label={`Shot ${currentShot.id} prompt`} rows="3" value={currentShot.prompt} onChange={(event) => updateDraft((next) => { next.productionPlan.shots[index].prompt = event.target.value; })} /></label>
                  <label>Beat<input aria-label={`Shot ${currentShot.id} beat`} value={currentShot.beat} onChange={(event) => updateDraft((next) => { next.productionPlan.shots[index].beat = event.target.value; })} /></label>
                  <label>Framing<input aria-label={`Shot ${currentShot.id} framing`} value={currentShot.framing} onChange={(event) => updateDraft((next) => { next.productionPlan.shots[index].framing = event.target.value; })} /></label>
                  <label>Duration (seconds)<input aria-label={`Shot ${currentShot.id} duration`} min="0.1" step="0.0001" type="number" value={currentShot.targetDurationSeconds} onChange={(event) => updateDraft((next) => { next.productionPlan.shots[index].targetDurationSeconds = Number(event.target.value); })} /></label>
                  <label>Start state<input aria-label={`Shot ${currentShot.id} start state`} value={currentShot.startState} onChange={(event) => updateDraft((next) => { next.productionPlan.shots[index].startState = event.target.value; })} /></label>
                  <label>End state<input aria-label={`Shot ${currentShot.id} end state`} value={currentShot.endState} onChange={(event) => updateDraft((next) => { next.productionPlan.shots[index].endState = event.target.value; })} /></label>
                  <label className="ve-film-prompt">Dialogue<textarea aria-label={`Shot ${currentShot.id} dialogue`} rows="2" value={currentShot.dialogue ?? ""} onChange={(event) => updateDraft((next) => { next.productionPlan.shots[index].dialogue = event.target.value || undefined; })} /></label>
                </div>
              </article>
            ))}
            <div className="ve-film-form">
              <label>Video model<input value={draft.productionPlan.model.id} onChange={(event) => updateDraft((next) => { next.productionPlan.model.id = event.target.value; })} /></label>
            </div>
          </section>
          <div className="ve-film-actions">
            <button disabled={busy} onClick={() => saveDraft().then(() => setNotice("Draft saved."), (error) => setNotice(error.message))} type="button">Save draft</button>
            <button className="ve-generate" disabled={busy} onClick={startRun} type="button">Render shot</button>
          </div>
        </div>
      ) : null}
      {notice ? <p aria-live="polite" className="ve-notice">{notice}</p> : null}
    </details>
  );
}
