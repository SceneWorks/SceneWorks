import React from "react";

const QWEN_MODEL_ID = "film_planner_qwen3_6_27b";

export function FilmPlanning({ draft, availability, operation, disabled, onChange, onStart, onCancel, onApply, onInstall }) {
  const planning = draft.planning ?? { provider: "prompt_refiner", thinkingMode: "disabled", refinePrompts: false };
  const qwen = availability?.providers?.find((item) => item.modelId === QWEN_MODEL_ID);
  const active = operation && ["running", "canceling"].includes(operation.status);
  const candidate = operation?.candidatePlan;
  function setProvider(provider) {
    onChange((next) => {
      next.planning = {
        ...next.planning,
        provider,
        modelId: provider === "native" ? QWEN_MODEL_ID : undefined,
        thinkingMode: provider === "native" ? "enabled" : "disabled",
      };
    });
  }
  return (
    <section aria-labelledby="film-planning-heading" className="ve-film-section">
      <h3 id="film-planning-heading">Planning</h3>
      <div className="ve-film-grid">
        <label>Planning provider
          <select aria-label="Planning provider" disabled={disabled || active} value={planning.provider} onChange={(event) => setProvider(event.target.value)}>
            <option value="prompt_refiner">Built-in prompt refiner (default)</option>
            <option value="native">Native Qwen3.6-27B (optional)</option>
          </select>
        </label>
        <label>Video model<input aria-label="Planning target video model" disabled value={draft.productionPlan.model.id} /></label>
        {planning.provider === "native" ? (
          <label>Thinking
            <select aria-label="Planner thinking mode" disabled={disabled || active} value={planning.thinkingMode ?? "enabled"} onChange={(event) => onChange((next) => { next.planning.thinkingMode = event.target.value; })}>
              <option value="enabled">Enabled (stored separately)</option>
              <option value="disabled">Disabled</option>
              <option value="auto">Model default</option>
            </select>
          </label>
        ) : null}
      </div>
      <p className="ve-film-provider-state">
        {planning.provider === "native"
          ? (qwen?.available ? "Qwen3.6-27B is installed and available." : "Qwen3.6-27B is not installed. The built-in planner remains available and no download starts automatically.")
          : "New drafts use the built-in prompt refiner. Qwen3.6-27B is not required."}
      </p>
      {planning.provider === "native" && !qwen?.available ? (
        <button disabled={disabled} onClick={() => onInstall(QWEN_MODEL_ID)} type="button">Install Qwen3.6-27B (large download)</button>
      ) : null}
      <div className="ve-film-actions">
        <button className="ve-generate" disabled={disabled || active || !draft.originalScript?.trim() || (planning.provider === "native" && !qwen?.available)} onClick={onStart} type="button">Generate candidate plan</button>
        {active ? <button disabled={operation.status === "canceling"} onClick={onCancel} type="button">Cancel planning</button> : null}
      </div>
      {operation ? (
        <div aria-live="polite" className={`ve-film-operation is-${operation.status}`}>
          <strong>{operation.status === "ready" ? "Candidate ready" : `Planning: ${operation.status}`}</strong>
          <span>{operation.detail}</span>
          {Number.isFinite(operation.progress) ? <progress aria-label="Planning progress" max="1" value={operation.progress} /> : null}
          <span>Planner: {operation.plannerModel}; target video model: {operation.videoModelId}</span>
          {operation.executions?.length ? <span>Execution: {operation.executions.map((item) => `${item.backend ?? "native"} / ${item.model}`).join(", ")}</span> : null}
          {operation.findings?.map((finding, index) => <p className="ve-film-finding" key={`${finding.field}-${index}`}>{finding.field}: {finding.message}</p>)}
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
