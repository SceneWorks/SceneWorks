import React, { useEffect, useState } from "react";
import { apiFetch } from "../../api.js";
import { useAppStatic } from "../../context/AppContext.js";

export function FilmWorkspace() {
  const { activeProject, token, refreshTimelines, setSelectedTimelineId } = useAppStatic();
  const [drafts, setDrafts] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [draft, setDraft] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");

  useEffect(() => {
    let canceled = false;
    setDrafts([]);
    setDraft(null);
    setSelectedId("");
    setNotice("");
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

  async function startRun() {
    setBusy(true);
    setNotice("");
    try {
      const saved = await saveDraft();
      const created = await apiFetch(`/api/v1/projects/${activeProject.id}/films/${saved.id}/runs`, token, { method: "POST" });
      let run = await apiFetch(`/api/v1/projects/${activeProject.id}/film-runs/${created.locator.id}/start`, token, { method: "POST" });
      setNotice("Rendering the shot in this project.");
      while (run.controllerActive) {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        run = await apiFetch(`/api/v1/projects/${activeProject.id}/film-runs/${created.locator.id}`, token);
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
        <div className="ve-film-form">
          <label>Film title<input value={draft.title} onChange={(event) => updateDraft((next) => { next.title = event.target.value; })} /></label>
          <label className="ve-film-prompt">Shot prompt<textarea aria-label="Shot prompt" rows="3" value={shot.prompt} onChange={(event) => updateDraft((next) => { next.productionPlan.shots[0].prompt = event.target.value; })} /></label>
          <label>Beat<input value={shot.beat} onChange={(event) => updateDraft((next) => { next.productionPlan.shots[0].beat = event.target.value; })} /></label>
          <label>Framing<input value={shot.framing} onChange={(event) => updateDraft((next) => { next.productionPlan.shots[0].framing = event.target.value; })} /></label>
          <label>Duration (seconds)<input min="0.1" step="0.0001" type="number" value={shot.targetDurationSeconds} onChange={(event) => updateDraft((next) => { next.productionPlan.shots[0].targetDurationSeconds = Number(event.target.value); })} /></label>
          <label>Video model<input value={draft.productionPlan.model.id} onChange={(event) => updateDraft((next) => { next.productionPlan.model.id = event.target.value; })} /></label>
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
