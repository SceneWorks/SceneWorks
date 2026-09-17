import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch } from "../../api.js";
import { assetCanRenderAsVideo, assetDisplayUrl, posterUrl } from "../assetMedia.jsx";
import {
  analyzeFilmTake,
  decideFilmTake,
  getFilmReview,
  repairFilmTake,
  replaceFilmTake,
  swapFilmTake,
} from "../../api/filmReview.js";

const TOPICS = ["identity", "object_state", "action_completion", "spatial_continuity", "cut_continuity"];
const FRAME_SCOPES = ["first", "last", "all", "any"];

function list(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function defaultQuestion(shot) {
  return {
    id: `${shot.id}_action`,
    topic: "action_completion",
    intended: shot.endState || shot.beat || "Intended end state",
    ask: "Does the final frame show the authored action completed? Answer yes or no.",
    expect: ["yes"],
    contradict: ["no"],
    frames: "last",
    mustObserve: true,
    acrossCut: false,
  };
}

export function normalizedPlan(draft) {
  const current = draft.reviewPlan ?? {};
  const shots = current.shots && typeof current.shots === "object" ? structuredClone(current.shots) : {};
  for (const shot of draft.productionPlan.shots) {
    if (!(shot.id in shots)) shots[shot.id] = { questions: [defaultQuestion(shot)] };
  }
  return {
    schemaVersion: current.schemaVersion ?? 1,
    id: current.id ?? `${draft.id}-review`,
    version: current.version ?? 1,
    description: current.description || "",
    sampling: { positions: current.sampling?.positions ?? [0.1, 0.5, 0.9] },
    limits: {
      maxSeconds: current.limits?.maxSeconds ?? 120,
      maxFramesPerShot: current.limits?.maxFramesPerShot ?? 3,
      maxQuestionsPerShot: current.limits?.maxQuestionsPerShot ?? 8,
      maxAnswerSeconds: current.limits?.maxAnswerSeconds ?? 30,
      maxNewTokens: current.limits?.maxNewTokens ?? 192,
      maxMemoryGb: current.limits?.maxMemoryGb ?? 16,
    },
    shots,
    uncertainBelow: Number.isFinite(current.uncertainBelow) ? current.uncertainBelow : 0.5,
  };
}

export function FilmReview({ active = true, draft, onChange, projectId, refreshTimelines, runControllerActive = false, runLocatorId = "", setNotice, setSelectedTimelineId, token }) {
  const [runId, setRunId] = useState("");
  const [view, setView] = useState(null);
  const [pending, setPending] = useState(false);
  const [reasons, setReasons] = useState({});
  const refreshRequest = useRef(0);
  const plan = useMemo(() => normalizedPlan(draft), [draft]);

  const refresh = useCallback(async () => {
    if (!projectId || !draft.id) return;
    const requestId = ++refreshRequest.current;
    try {
      let locatorId = runLocatorId;
      if (!locatorId) {
        const runs = await apiFetch(`/api/v1/projects/${projectId}/film-runs`, token);
        const latest = runs.find((item) => item.locator.draftId === draft.id && item.record);
        if (refreshRequest.current !== requestId) return;
        if (!latest) {
          setRunId("");
          setView(null);
          return;
        }
        locatorId = latest.locator.id;
      }
      const next = await getFilmReview(projectId, locatorId, token);
      if (refreshRequest.current !== requestId) return;
      setRunId(locatorId);
      setView(next);
    } catch (error) {
      if (refreshRequest.current === requestId) setNotice(error.message);
    }
  }, [draft.id, projectId, runLocatorId, setNotice, token]);

  // The three workspace panels stay mounted to preserve unsaved edits. Refresh when Review becomes
  // visible or its durable controller transitions, so Resume cannot leave a terminal cached view.
  useEffect(() => { if (active) refresh(); }, [active, refresh, runControllerActive]);
  const pollingActive = Boolean(runControllerActive || view?.run?.controllerActive);
  useEffect(() => {
    if (!active || !pollingActive) return undefined;
    let canceled = false;
    let timer = null;
    async function poll() {
      await refresh();
      if (!canceled) timer = window.setTimeout(poll, 800);
    }
    timer = window.setTimeout(poll, 800);
    return () => {
      canceled = true;
      refreshRequest.current += 1;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [active, pollingActive, refresh]);

  function changePlan(mutator) {
    onChange((next) => {
      next.reviewPlan = normalizedPlan(next);
      mutator(next.reviewPlan);
    });
  }

  async function mutate(action, message) {
    if (!runId) return;
    setPending(true);
    try {
      const next = await action();
      setView(next);
      setNotice(message);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setPending(false);
    }
  }

  async function openTimeline(timelineId) {
    if (!timelineId) return;
    await refreshTimelines(projectId);
    setSelectedTimelineId(timelineId);
  }

  const disabledReason = view?.actionDisabledReason || (pending ? "A review action is being saved." : "");
  const assetsById = new Map((view?.takeAssets ?? []).map((asset) => [asset.id, asset]));
  const observationsByShot = new Map();
  for (const observation of view?.observations ?? []) {
    const items = observationsByShot.get(observation.shotId) ?? [];
    items.push(observation);
    observationsByShot.set(observation.shotId, items);
  }

  return (
    <section aria-labelledby="film-review-heading" className="ve-film-section ve-film-review">
      <div className="ve-film-section-heading">
        <div>
          <h3 id="film-review-heading">Review</h3>
          <p>Runs pin questions. Findings stay advisory.</p>
        </div>
      </div>

      <details>
        <summary>Review questions, frames, and limits</summary>
        <fieldset className="ve-film-review-plan">
          <legend>Future-run review plan</legend>
          <label>Description<input value={plan.description} onChange={(event) => changePlan((next) => { next.description = event.target.value; })} /></label>
          <label>Frame positions (0–1, comma separated)<input aria-label="Review frame positions" value={plan.sampling.positions.join(", ")} onChange={(event) => changePlan((next) => { next.sampling.positions = list(event.target.value).map(Number); })} /></label>
          {[
            ["maxSeconds", "Maximum seconds", 1], ["maxFramesPerShot", "Maximum frames per shot", 1],
            ["maxQuestionsPerShot", "Maximum questions per shot", 1], ["maxAnswerSeconds", "Maximum answer seconds", 1],
            ["maxNewTokens", "Maximum answer tokens", 1], ["maxMemoryGb", "Maximum memory (GB)", 0.1],
          ].map(([field, label, step]) => <label key={field}>{label}<input min={step} step={step} type="number" value={plan.limits[field]} onChange={(event) => changePlan((next) => { next.limits[field] = Number(event.target.value); })} /></label>)}
          <label>Uncertain below<input aria-label="Review uncertainty threshold" max="0.99" min="0" step="0.01" type="number" value={plan.uncertainBelow} onChange={(event) => changePlan((next) => { next.uncertainBelow = Number(event.target.value); })} /></label>
        </fieldset>
        {Object.entries(plan.shots).map(([shotId, spec]) => (
          <article className="ve-film-review-questions" key={shotId}>
            <header><strong>{shotId}</strong><button onClick={() => changePlan((next) => next.shots[shotId].questions.push({ ...defaultQuestion(draft.productionPlan.shots.find((shot) => shot.id === shotId) ?? {}), id: `question-${next.shots[shotId].questions.length + 1}` }))} type="button">Add question</button></header>
            {spec.questions.map((question, index) => (
              <fieldset key={`${question.id}-${index}`}>
                <legend>Question {index + 1}</legend>
                <label>ID<input value={question.id} onChange={(event) => changePlan((next) => { next.shots[shotId].questions[index].id = event.target.value; })} /></label>
                <label>Topic<select value={question.topic} onChange={(event) => changePlan((next) => { next.shots[shotId].questions[index].topic = event.target.value; })}>{TOPICS.map((topic) => <option key={topic}>{topic}</option>)}</select></label>
                <label>Intended state<textarea value={question.intended} onChange={(event) => changePlan((next) => { next.shots[shotId].questions[index].intended = event.target.value; })} /></label>
                <label>Question<textarea aria-label={`${shotId} review question ${index + 1}`} value={question.ask} onChange={(event) => changePlan((next) => { next.shots[shotId].questions[index].ask = event.target.value; })} /></label>
                <label>Matching answers<input value={question.expect.join(", ")} onChange={(event) => changePlan((next) => { next.shots[shotId].questions[index].expect = list(event.target.value); })} /></label>
                <label>Contradicting answers<input value={question.contradict.join(", ")} onChange={(event) => changePlan((next) => { next.shots[shotId].questions[index].contradict = list(event.target.value); })} /></label>
                <label>Frames<select aria-label={`${shotId} question ${index + 1} frames`} value={question.frames} onChange={(event) => changePlan((next) => { next.shots[shotId].questions[index].frames = event.target.value; })}>{FRAME_SCOPES.map((scope) => <option key={scope}>{scope}</option>)}</select></label>
                <label><input checked={Boolean(question.mustObserve)} onChange={(event) => changePlan((next) => { next.shots[shotId].questions[index].mustObserve = event.target.checked; })} type="checkbox" />Unobserved is actionable</label>
                <label><input checked={Boolean(question.acrossCut)} onChange={(event) => changePlan((next) => { next.shots[shotId].questions[index].acrossCut = event.target.checked; })} type="checkbox" />Compare across cut</label>
                <button disabled={spec.questions.length === 1} onClick={() => changePlan((next) => { next.shots[shotId].questions.splice(index, 1); })} type="button">Remove question</button>
              </fieldset>
            ))}
          </article>
        ))}
      </details>

      {!view ? <p>Render a shot to review takes.</p> : (
        <div className="ve-film-review-runs">
          <div className="ve-film-actions">
            <button disabled={Boolean(disabledReason)} onClick={() => mutate(() => analyzeFilmTake(projectId, runId, [], token), "Assistive review started.")} title={disabledReason} type="button">Analyze selected takes</button>
            <button disabled={!view.run.record?.timeline?.timelineId} onClick={() => openTimeline(view.run.record.timeline.timelineId)} type="button">Open saved cut</button>
            <button disabled={!view.reviewTimelineId} onClick={() => openTimeline(view.reviewTimelineId)} type="button">Open review frames</button>
          </div>
          {disabledReason ? <p className="ve-film-warning" role="status">Actions unavailable: {disabledReason}</p> : null}
          {["failed", "rejected"].includes(view.reviewOperation?.status) ? <p role="alert">Assistive review failed: {view.reviewOperation.detail}. Correct the problem and analyze again; existing observations may be partial.</p> : null}
          {view.reviewOperation?.status === "running" && !view.run.controllerActive ? <p role="alert">Assistive review was interrupted. Analyze again to complete it; existing observations may be partial.</p> : null}
          <p>{view.assistiveNotice}</p>
          {(view.run.record?.shots ?? []).map((shot) => {
            const selection = view.selections.find((item) => item.shotId === shot.shotId);
            const reviews = observationsByShot.get(shot.shotId) ?? [];
            const repairDisabled = Boolean(disabledReason || selection?.trimConflict) || !shot.attempts.some((attempt) => attempt.take);
            const repairTitle = disabledReason || (selection?.trimConflict ? "Resolve the trim conflict first." : "");
            return (
              <article className="ve-film-review-shot" key={shot.shotId}>
                <header><h4>{shot.shotId}</h4><span className={`ve-film-selection-state ${selection?.state ?? ""}`}>{selection?.state?.replaceAll("_", " ") || "selection unavailable"}</span></header>
                {selection?.trimConflict ? <p className="ve-film-warning">Resolve the trim conflict before using the pending take. Current saved asset: {selection.timelineAssetId || "none"}.</p> : null}
                {shot.needsReview?.length ? <ul className="ve-film-findings">{shot.needsReview.map((flag, index) => <li key={`${flag.raisedAt}-${index}`}>Dependency stale from {flag.sourceShotId} ({flag.dependency}): {flag.reason}</li>)}</ul> : null}
                <div className="ve-film-take-grid">
                  {shot.attempts.filter((attempt) => attempt.take).map((attempt) => {
                    const asset = assetsById.get(attempt.take.assetId);
                    const isGenerationSelected = shot.selectedAttempt === attempt.attempt;
                    const isSavedCut = selection?.timelineAssetId === attempt.take.assetId;
                    const canDecide = isGenerationSelected && isSavedCut && !selection?.trimConflict;
                    const decisionTitle = disabledReason || (!canDecide ? "Use this take in the saved cut first." : "");
                    return (
                      <article className="ve-film-take" key={attempt.attempt}>
                        <header><strong>Attempt {attempt.attempt}</strong><span>{attempt.rejection ? "Rejected" : isSavedCut ? "Current saved cut" : isGenerationSelected ? "Generation selection" : "History"}</span></header>
                        {asset && assetCanRenderAsVideo(asset) ? <video aria-label={`${shot.shotId} attempt ${attempt.attempt} preview`} controls poster={posterUrl(asset)} preload="metadata" src={assetDisplayUrl(asset)} /> : <div className="ve-film-take-missing">Preview unavailable · asset {attempt.take.assetId}</div>}
                        <dl><dt>Asset</dt><dd>{attempt.take.assetId}</dd><dt>Job</dt><dd>{attempt.jobId || "unknown"}</dd><dt>Model</dt><dd>{attempt.take.model}</dd><dt>Seed</dt><dd>{attempt.take.seed ?? "random"}</dd></dl>
                        {attempt.rejection ? <p>Rejected {attempt.rejection.at}: {attempt.rejection.reason}</p> : null}
                        <div className="ve-film-actions">
                          <button disabled={!canDecide || Boolean(disabledReason)} onClick={() => mutate(() => decideFilmTake(projectId, runId, shot.shotId, "accept", reasons[shot.shotId] || "Human accepted.", token), `${shot.shotId} accepted.`)} title={decisionTitle} type="button">Accept</button>
                          <button disabled={!canDecide || Boolean(disabledReason)} onClick={() => mutate(() => decideFilmTake(projectId, runId, shot.shotId, "reject", reasons[shot.shotId] || "Human rejected.", token), `${shot.shotId} rejected; history kept.`)} title={decisionTitle} type="button">Reject</button>
                          <button disabled={Boolean(disabledReason) || (isSavedCut && isGenerationSelected)} onClick={() => mutate(() => swapFilmTake(projectId, runId, shot.shotId, attempt.take.assetId, token), `${shot.shotId} saved cut now uses attempt ${attempt.attempt}.`)} title={disabledReason || (isSavedCut && isGenerationSelected ? "Already selected in the saved cut." : "")} type="button">Use in saved cut</button>
                        </div>
                      </article>
                    );
                  })}
                </div>
                <label>Decision or repair reason<input aria-label={`${shot.shotId} review reason`} value={reasons[shot.shotId] ?? ""} onChange={(event) => setReasons((current) => ({ ...current, [shot.shotId]: event.target.value }))} /></label>
                <div className="ve-film-actions">
                  <button disabled={repairDisabled} onClick={() => mutate(() => replaceFilmTake(projectId, runId, shot.shotId, reasons[shot.shotId] || "Replacement requested.", token), `${shot.shotId} replacement started; history kept.`)} title={repairTitle} type="button">Render one replacement</button>
                  <button disabled={repairDisabled} onClick={() => mutate(() => repairFilmTake(projectId, runId, shot.shotId, reasons[shot.shotId] || "Repair requested.", token), `${shot.shotId} repair started; review required.`)} title={repairTitle} type="button">Repair from findings</button>
                </div>
                {reviews.map((review) => <section aria-label={`${shot.shotId} assistive findings`} className="ve-film-review-findings" key={review.reviewId}><strong>Review of attempt {review.attempt}</strong>{review.mismatches.length ? <ul>{review.mismatches.map((flag) => <li key={`${flag.questionId}-${flag.severity}`}><b>{flag.severity}</b> · {flag.topic}: intended “{flag.intended}”; observed “{flag.observed}” ({Math.round(flag.confidence * 100)}%). {flag.detail}</li>)}</ul> : <p>No advisory mismatches recorded.</p>}</section>)}
                {shot.humanDecision ? <p><strong>Human decision:</strong> {shot.humanDecision.state} attempt {shot.humanDecision.attempt} · {shot.humanDecision.reason}</p> : <p>No human decision.</p>}
              </article>
            );
          })}
          {view.run.record?.decisions?.length ? <details><summary>Decision provenance</summary><ol>{view.run.record.decisions.map((decision, index) => <li key={`${decision.at}-${index}`}>{decision.at} · {decision.action}{decision.shotId ? ` · ${decision.shotId}` : ""}: {decision.detail}</li>)}</ol></details> : null}
        </div>
      )}
    </section>
  );
}
