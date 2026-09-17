import React, { lazy, Suspense, useState } from "react";
import { errorStatuses, terminalStatuses } from "../../jobTypes.js";

const QWEN_MODEL_ID = "film_planner_qwen3_6_27b";
const FilmPlannerConnection = lazy(() => import("./FilmPlannerConnection.jsx"));

export function FilmPlanning({ draft, availability, operation, disabled, installError, installJob, models = [], onChange, onStart, onCancel, onApply, onInstall, onNotice, onRefreshAvailability, token }) {
  const [maxRepairRounds, setMaxRepairRounds] = useState("2");
  const planning = draft.planning ?? { provider: "prompt_refiner", thinkingMode: "disabled", refinePrompts: false };
  const qwen = availability?.providers?.find((item) => item.modelId === QWEN_MODEL_ID);
  const active = operation && ["running", "canceling"].includes(operation.status);
  const candidate = operation?.candidatePlan;
  const installActive = installJob && !terminalStatuses.has(installJob.status);
  const videoModels = models.filter((model) => model.type === "video" && model.usable !== false);
  const parsedMaxRepairRounds = Number(maxRepairRounds);
  const repairRoundsValid = Number.isInteger(parsedMaxRepairRounds) && parsedMaxRepairRounds >= 0 && parsedMaxRepairRounds <= 5;
  function setProvider(provider) {
    onChange((next) => {
      next.planning = {
        ...next.planning,
        provider,
        modelId: provider === "native" ? QWEN_MODEL_ID : undefined,
        connectionId: provider === "openai_compatible" ? next.planning?.connectionId : undefined,
        thinkingMode: provider === "native" ? "enabled" : "disabled",
        sendReferencePixels: provider === "openai_compatible" ? Boolean(next.planning?.sendReferencePixels) : false,
      };
    });
    if (provider === "native") void onRefreshAvailability?.();
  }
  return (
    <section aria-labelledby="film-planning-heading" className="ve-film-section">
      <h3 id="film-planning-heading">Planning</h3>
      <div className="ve-film-grid">
        <label>Planning provider
          <select aria-label="Planning provider" disabled={disabled || active} value={planning.provider} onChange={(event) => setProvider(event.target.value)}>
            <option value="prompt_refiner">Built-in prompt refiner (default)</option>
            <option value="native">Native Qwen3.6-27B (optional)</option>
            <option value="openai_compatible">Saved OpenAI-compatible connection</option>
          </select>
        </label>
        <label>Video model{videoModels.length ? <select aria-label="Planning target video model" disabled={disabled || active} value={draft.productionPlan.model.id} onChange={(event) => onChange((next) => { next.productionPlan.model.id = event.target.value; })}>{videoModels.map((model) => <option key={model.id} value={model.id}>{model.name ?? model.id}{model.installState === "missing" ? " (not installed)" : ""}</option>)}</select> : <input aria-label="Planning target video model" disabled={disabled || active} value={draft.productionPlan.model.id} onChange={(event) => onChange((next) => { next.productionPlan.model.id = event.target.value; })} />}</label>
      </div>
      {planning.provider === "openai_compatible" ? (
        <Suspense fallback={<p>Loading connection settings…</p>}>
          <FilmPlannerConnection active={active} disabled={disabled} onChange={onChange} onNotice={onNotice} planning={planning} token={token} />
        </Suspense>
      ) : null}
      <details className="ve-film-advanced">
        <summary>Advanced planning settings</summary>
        <div className="ve-film-grid">
          {planning.provider !== "prompt_refiner" ? (
            <label>Thinking
              <select aria-label="Planner thinking mode" disabled={disabled || active} value={planning.thinkingMode ?? "enabled"} onChange={(event) => onChange((next) => { next.planning.thinkingMode = event.target.value; })}>
                <option value="enabled">Enabled (stored separately)</option>
                <option value="disabled">Disabled</option>
                <option value="auto">Model default</option>
              </select>
            </label>
          ) : null}
          <label className="ve-film-reference-check"><input checked={Boolean(planning.refinePrompts)} disabled={disabled || active} onChange={(event) => onChange((next) => { next.planning.refinePrompts = event.target.checked; })} type="checkbox" /> Run model-specific prompt refinement when compiling shots</label>
          <label>Maximum planner repair rounds<input aria-label="Maximum planner repair rounds" disabled={disabled || active} max="5" min="0" onChange={(event) => setMaxRepairRounds(event.target.value)} step="1" type="number" value={maxRepairRounds} /></label>
        </div>
      </details>
      <p className="ve-film-provider-state">
        {planning.provider === "native"
          ? (availability == null
            ? "Checking Qwen3.6-27B availability. No download starts automatically."
            : qwen?.available
              ? "Qwen3.6-27B is installed and available."
              : "Qwen3.6-27B is not installed. The built-in planner remains available and no download starts automatically.")
          : planning.provider === "openai_compatible"
            ? "External planning runs only when explicitly selected. Configured credentials never change the local planner route."
          : "New drafts use the built-in prompt refiner. Qwen3.6-27B is not required."}
      </p>
      {planning.provider === "native" && availability != null && !qwen?.available ? (
        <button disabled={disabled || installActive} onClick={() => onInstall(QWEN_MODEL_ID)} type="button">Install Qwen3.6-27B (large download)</button>
      ) : null}
      {installJob ? <div aria-live="polite" className={`ve-film-operation is-${installJob.status}`}>
        <strong>Qwen3.6-27B download: {installJob.status}</strong>
        {Number.isFinite(installJob.progress) ? <progress aria-label="Qwen3.6-27B download progress" max="1" value={installJob.progress} /> : null}
        {errorStatuses.has(installJob.status) && (installJob.message || installJob.error) ? <span>{installJob.message || installJob.error}</span> : null}
        {installError ? <span role="alert">Status refresh failed: {installError}. Retrying.</span> : null}
      </div> : null}
      <div className="ve-film-actions">
        <button className="ve-generate" disabled={disabled || active || !repairRoundsValid || !draft.originalScript?.trim() || (planning.provider === "native" && !qwen?.available) || (planning.provider === "openai_compatible" && (!planning.connectionId || !planning.modelId?.trim()))} onClick={() => onStart(parsedMaxRepairRounds)} type="button">Generate candidate plan</button>
        {active ? <button disabled={operation.status === "canceling"} onClick={onCancel} type="button">Cancel planning</button> : null}
      </div>
      {operation ? (
        <div aria-live="polite" className={`ve-film-operation is-${operation.status}`}>
          <strong>{operation.status === "ready" ? "Candidate ready" : `Planning: ${operation.status}`}</strong>
          <span>{operation.detail}</span>
          {Number.isFinite(operation.progress) ? <progress aria-label="Planning progress" max="1" value={operation.progress} /> : null}
          <span>Planner: {operation.plannerModel}; target video model: {operation.videoModelId}</span>
          {Number.isInteger(operation.maxRepairRounds) ? <span>Maximum repair rounds: {operation.maxRepairRounds}</span> : null}
          {operation.executions?.length ? <span>Execution: {operation.executions.map((item) => `${item.backend ?? "native"} / ${item.model}`).join(", ")}</span> : null}
          {operation.findings?.length ? (
            <ul aria-label="Planning findings" className="ve-film-planning-findings">
              {operation.findings.map((finding, index) => <li key={`${finding.shotId ?? "plan"}-${finding.field}-${index}`}>
                <strong>{finding.shotId ? `${finding.shotId} · ` : ""}{finding.field}:</strong>{" "}
                <span>{finding.message}</span>
              </li>)}
            </ul>
          ) : null}
        </div>
      ) : null}
      {candidate ? (
        <div className="ve-film-candidate">
          <h4>Generated shot plan</h4>
          {candidate.shots.map((shot) => <article key={shot.id}><strong>{shot.id}: {shot.beat}</strong><p>{shot.prompt}</p><small>{shot.framing} · {shot.targetDurationSeconds}s{shot.dialogue ? ` · ${shot.dialogue}` : ""}</small></article>)}
          <button disabled={disabled} onClick={onApply} type="button">Replace current edited plan with this candidate</button>
          <p>The replacement is explicit. Regeneration never overwrites the current plan or starts rendering.</p>
        </div>
      ) : null}
    </section>
  );
}
