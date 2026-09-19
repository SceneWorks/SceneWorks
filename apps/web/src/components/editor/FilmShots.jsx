import React, { useEffect, useRef, useState } from "react";
import { FILM_SHOT_STATE_LABELS, filmShotState } from "./filmShotState.js";

const CONDITIONING_MODES = ["text_to_video", "image_to_video", "first_last_frame", "reference_to_video"];
const DEPENDENCY_KINDS = ["conditioning", "continuity"];

function asList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function setOptional(target, key, value) {
  if (value === "") delete target[key];
  else target[key] = value;
}

function setOptionalNumber(target, key, value) {
  if (value === "") delete target[key];
  else target[key] = Number(value);
}

function nextShotId(shots) {
  const used = new Set(shots.map((shot) => shot.id));
  for (let number = 10; ; number += 10) {
    const id = `SH${String(number).padStart(3, "0")}`;
    if (!used.has(id)) return id;
  }
}

function newShot(shots) {
  const id = nextShotId(shots);
  return {
    id, beat: "New shot", framing: "wide", prompt: "", targetDurationSeconds: 5.1667,
    startState: "Opening state", endState: "Closing state",
    // Required by plan schema 3 (sc-24026), and blank like `prompt` is: the draft PUT decodes into
    // a Rust `Shot` where `audio` has no default, so omitting the key makes the draft unsaveable
    // with an anonymous decode error instead of the shot-named finding the Audio field shows.
    // Matches `FilmDraft::manual_one_shot`.
    audio: "",
    conditioning: { mode: "text_to_video", referenceRoles: [] },
    continuityRoles: [], dependsOn: [],
  };
}

function Finding({ findings, shotId, field }) {
  const matches = findings.filter((finding) => finding.shotId === shotId
    && (!field || finding.field === field || finding.field.startsWith(`${field}.`) || finding.field.startsWith(`${field}[`)));
  if (!matches.length) return null;
  return <ul className="ve-film-findings">{matches.map((finding, index) => <li key={`${finding.field}-${index}`}>{finding.message}</li>)}</ul>;
}

function NumberInput({ label, value, onChange, min, step = "1", list, options = [] }) {
  return <label>{label}<input list={list} min={min} step={step} type="number" value={value ?? ""} onChange={(event) => onChange(event.target.value)} />{list ? <datalist id={list}>{options.map((option) => <option key={option} value={option} />)}</datalist> : null}</label>;
}

function exportJson(name, value) {
  const url = URL.createObjectURL(new Blob([`${JSON.stringify(value, null, 2)}\n`], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function FilmShots({ capabilities, compiled, disabled, draft, findings = [], models = [], onChange, onImportError, run = null, selectedShotIds, setSelectedShotIds }) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const planInput = useRef(null);
  const compiledInput = useRef(null);
  const shots = draft.productionPlan.shots;
  const shot = shots[Math.min(selectedIndex, shots.length - 1)];
  // Every approved role, image-backed or DESCRIBED-ONLY (sc-24025). These belong in
  // `continuityRoles` — the only slot a role with no image is allowed in.
  const approvedRoles = draft.referencePack.references.filter((entry) => entry.approved).map((entry) => entry.role);
  // The approved roles a CONDITIONING slot may name: a described-only role supplies no image, and
  // `validate_plan_against_pack` refuses one in the keyframe slots and in `referenceRoles` alike,
  // so offering it here would build a draft the server rejects.
  const conditionableRoles = draft.referencePack.references.filter((entry) => entry.approved && Boolean(entry.file)).map((entry) => entry.role);
  const modes = capabilities?.modes?.length ? capabilities.modes : CONDITIONING_MODES;
  const resolutions = capabilities?.resolutions ?? [];
  const turboLoras = capabilities?.turboLoras ?? [];
  const videoModels = models.filter((model) => model.type === "video" && model.usable !== false);
  const selectedModel = videoModels.find((model) => model.id === draft.productionPlan.model.id);
  const tiers = selectedModel?.variants?.map((variant) => variant.variant)
    ?? selectedModel?.mlxTiers ?? [];

  function mutateShot(mutator) {
    onChange((next) => mutator(next.productionPlan.shots[Math.min(selectedIndex, next.productionPlan.shots.length - 1)]));
  }

  function move(delta) {
    const destination = selectedIndex + delta;
    if (destination < 0 || destination >= shots.length) return;
    onChange((next) => {
      const [moved] = next.productionPlan.shots.splice(selectedIndex, 1);
      next.productionPlan.shots.splice(destination, 0, moved);
    });
    setSelectedIndex(destination);
  }

  async function importDocument(event, kind) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      const document = JSON.parse(await file.text());
      if (kind === "plan" && (!Array.isArray(document.shots) || !document.model)) throw new Error("Production plan must contain model and shots.");
      if (kind === "compiled" && (!Array.isArray(document.requests) || !document.planSha256)) throw new Error("Compiled document must contain planSha256 and requests.");
      onChange((next) => {
        if (kind === "plan") {
          next.productionPlan = document;
          next.renderRegime = "custom";
        }
        else next.compiledPlan = document;
      });
      if (kind === "plan") {
        setSelectedIndex(0);
        setSelectedShotIds(document.shots.map((item) => item.id));
      }
    } catch (error) { onImportError(error.message); }
  }

  const advanced = draft.productionPlan.model.advanced ?? {};
  const advancedFindings = findings.filter((finding) => finding.shotId == null && (finding.field === "model" || finding.field.startsWith("model.") || finding.field === "limits" || finding.field.startsWith("limits.")));
  useEffect(() => {
    if (advancedFindings.length) setAdvancedOpen(true);
  }, [advancedFindings.length]);
  return (
    <section aria-labelledby="film-shots-heading" className="ve-film-section ve-film-shots">
      <div className="ve-film-section-heading">
        <div><h3 id="film-shots-heading">Shots and render controls</h3><p>Choose which ordered shots to render, then inspect the exact compiled requests before dispatch.</p></div>
        <div className="ve-film-actions">
          <button disabled={disabled} onClick={() => { onChange((next) => next.productionPlan.shots.push(newShot(next.productionPlan.shots))); setSelectedIndex(shots.length); }} type="button">Add shot</button>
          <button disabled={disabled} onClick={() => exportJson(`${draft.id}-production-plan.json`, draft.productionPlan)} type="button">Export production plan</button>
          <button disabled={disabled} onClick={() => planInput.current?.click()} type="button">Import production plan</button>
          <input accept="application/json,.json" aria-label="Production plan file" hidden onChange={(event) => importDocument(event, "plan")} ref={planInput} type="file" />
          <button disabled={disabled || !draft.compiledPlan} onClick={() => exportJson(`${draft.id}-compiled.json`, draft.compiledPlan)} type="button">Export compiled plan</button>
          <button disabled={disabled} onClick={() => compiledInput.current?.click()} type="button">Import compiled plan</button>
          <input accept="application/json,.json" aria-label="Compiled plan file" hidden onChange={(event) => importDocument(event, "compiled")} ref={compiledInput} type="file" />
          {draft.compiledPlan ? <button disabled={disabled} onClick={() => onChange((next) => { delete next.compiledPlan; })} type="button">Use authored prompts</button> : null}
        </div>
      </div>

      <details className="ve-film-advanced ve-film-plan-controls" onToggle={(event) => setAdvancedOpen(event.currentTarget.open)} open={advancedOpen}>
        <summary>Advanced video and budget settings{advancedFindings.length ? ` · ${advancedFindings.length} finding${advancedFindings.length === 1 ? "" : "s"}` : ""}</summary>
        <fieldset disabled={disabled}>
          <legend>Video and budgets</legend>
          <div className="ve-film-form">
          <label>Video model{videoModels.length ? <select aria-label="Video model" value={draft.productionPlan.model.id} onChange={(event) => onChange((next) => { next.productionPlan.model.id = event.target.value; })}>{videoModels.map((model) => <option key={model.id} value={model.id}>{model.name ?? model.id}{model.installState === "missing" ? " (not installed)" : ""}</option>)}</select> : <input aria-label="Video model" value={draft.productionPlan.model.id} onChange={(event) => onChange((next) => { next.productionPlan.model.id = event.target.value; })} />}</label>
          <label>Tier{tiers.length ? <select aria-label="Video model tier" value={draft.productionPlan.model.tier ?? ""} onChange={(event) => onChange((next) => setOptional(next.productionPlan.model, "tier", event.target.value))}><option value="">Model default</option>{tiers.map((tier) => <option key={tier}>{tier}</option>)}</select> : <input aria-label="Video model tier" value={draft.productionPlan.model.tier ?? ""} onChange={(event) => onChange((next) => setOptional(next.productionPlan.model, "tier", event.target.value))} />}</label>
          <NumberInput label="Frames per second" min="1" onChange={(value) => onChange((next) => setOptionalNumber(next.productionPlan.model, "fps", value))} value={draft.productionPlan.model.fps} />
          <label>Default resolution<input aria-label="Default resolution" list="film-resolution-options" value={draft.productionPlan.model.resolution ?? ""} onChange={(event) => onChange((next) => setOptional(next.productionPlan.model, "resolution", event.target.value))} /></label>
          <datalist id="film-resolution-options">{resolutions.map((value) => <option key={value} value={value} />)}</datalist>
          <label>Adapters (comma separated)<input aria-label="Plan adapters" list="film-adapter-options" value={(draft.productionPlan.model.loras ?? []).join(", ")} onChange={(event) => onChange((next) => { next.renderRegime = "custom"; next.productionPlan.model.loras = asList(event.target.value); })} /></label>
          <datalist id="film-adapter-options">{turboLoras.map((lora) => <option key={lora.id} value={lora.id}>{lora.name}</option>)}</datalist>
          <NumberInput label="Steps" min="1" onChange={(value) => onChange((next) => { next.renderRegime = "custom"; next.productionPlan.model.advanced ??= {}; setOptionalNumber(next.productionPlan.model.advanced, "steps", value); })} value={advanced.steps} />
          <NumberInput label="Reference image short edge" max="2048" min="1024" onChange={(value) => onChange((next) => { next.productionPlan.model.advanced ??= {}; setOptionalNumber(next.productionPlan.model.advanced, "referenceImageShortEdge", value); })} value={advanced.referenceImageShortEdge} />
          <NumberInput label="Maximum run seconds" min="1" onChange={(value) => onChange((next) => { next.productionPlan.limits.maxRunSeconds = Number(value); })} value={draft.productionPlan.limits.maxRunSeconds} />
          <NumberInput label="Maximum shot seconds" min="1" onChange={(value) => onChange((next) => { next.productionPlan.limits.maxShotSeconds = Number(value); })} value={draft.productionPlan.limits.maxShotSeconds} />
          <NumberInput label="Attempts per shot" min="1" onChange={(value) => onChange((next) => { next.productionPlan.limits.maxAttemptsPerShot = Number(value); })} value={draft.productionPlan.limits.maxAttemptsPerShot} />
          <NumberInput label="Execution memory budget (GB)" min="0.1" step="0.1" onChange={(value) => onChange((next) => { next.productionPlan.limits.maxMemoryGb = Number(value); })} value={draft.productionPlan.limits.maxMemoryGb} />
          <NumberInput label="Planner memory budget (GB)" min="0.1" step="0.1" onChange={(value) => onChange((next) => setOptionalNumber(next.productionPlan.limits, "plannerMaxMemoryGb", value))} value={draft.productionPlan.limits.plannerMaxMemoryGb} />
          </div>
          <Finding field="model" findings={findings} /><Finding field="limits" findings={findings} />
          {capabilities ? <p className="ve-film-capability-summary">Available modes: {capabilities.modes.join(", ") || "none"}. Durations: {capabilities.durations.join(", ") || "model-defined"}. Reference slots: {capabilities.maxReferenceImages}. Installed compatible adapters: {capabilities.turboLoras.map((item) => `${item.name} (${item.steps} steps)`).join(", ") || "none"}.</p> : null}
        </fieldset>
      </details>

      <div className="ve-film-shot-layout">
        <ol aria-label="Ordered shots" className="ve-film-shot-list">
          {shots.map((item, index) => (
            <li className={index === selectedIndex ? "selected" : ""} key={item.id}>
              <label><input aria-label={`Render ${item.id}`} checked={selectedShotIds.includes(item.id)} onChange={(event) => setSelectedShotIds((ids) => event.target.checked ? [...new Set([...ids, item.id])] : ids.filter((id) => id !== item.id))} type="checkbox" />Render</label>
              <button aria-pressed={index === selectedIndex} onClick={() => setSelectedIndex(index)} type="button"><strong>{item.id}</strong><span>{item.beat || "Untitled shot"}</span></button>
              {run ? <em className={`ve-film-shot-state ${filmShotState(item.id, run)}`}>{FILM_SHOT_STATE_LABELS[filmShotState(item.id, run)]}</em> : null}
            </li>
          ))}
        </ol>
        {shot ? (
          <article className="ve-film-shot-inspector">
            <header><h4>{shot.id}</h4><div className="ve-film-actions"><button disabled={disabled || selectedIndex === 0} onClick={() => move(-1)} type="button">Move up</button><button disabled={disabled || selectedIndex === shots.length - 1} onClick={() => move(1)} type="button">Move down</button><button disabled={disabled || shots.length === 1} onClick={() => { const removed = shot.id; onChange((next) => { next.productionPlan.shots.splice(selectedIndex, 1); next.productionPlan.shots.forEach((candidate) => { candidate.dependsOn = (candidate.dependsOn ?? []).filter((dependency) => dependency.shotId !== removed); if (candidate.conditioning.chainFromShotId === removed) delete candidate.conditioning.chainFromShotId; }); }); setSelectedShotIds((ids) => ids.filter((id) => id !== removed)); setSelectedIndex(Math.max(0, selectedIndex - 1)); }} type="button">Remove</button></div></header>
            <fieldset className="ve-film-inspector-fields" disabled={disabled}>
              <legend>Shot intent and request</legend>
              <label>Shot ID<input aria-label="Shot ID" value={shot.id} onChange={(event) => { const before = shot.id; const after = event.target.value; onChange((next) => { next.productionPlan.shots[selectedIndex].id = after; if (next.reviewPlan?.shots?.[before] && !next.reviewPlan.shots[after]) { next.reviewPlan.shots[after] = next.reviewPlan.shots[before]; delete next.reviewPlan.shots[before]; } }); setSelectedShotIds((ids) => ids.map((id) => id === before ? after : id)); }} /></label>
              <label>Beat ID<input value={shot.beatId ?? ""} onChange={(event) => mutateShot((next) => setOptional(next, "beatId", event.target.value))} /></label>
              <label>Beat / action<textarea aria-label={`Shot ${shot.id} beat`} rows="2" value={shot.beat} onChange={(event) => mutateShot((next) => { next.beat = event.target.value; })} /></label>
              <label>Framing<input value={shot.framing} onChange={(event) => mutateShot((next) => { next.framing = event.target.value; })} /></label>
              <label className="ve-film-prompt">Prompt<textarea aria-label={`Shot ${shot.id} prompt`} rows="4" value={shot.prompt} onChange={(event) => mutateShot((next) => { next.prompt = event.target.value; })} /></label>
              <label className="ve-film-prompt">Negative prompt<textarea value={shot.negativePrompt ?? ""} onChange={(event) => mutateShot((next) => setOptional(next, "negativePrompt", event.target.value))} /></label>
              <NumberInput label="Duration (seconds)" list="film-duration-options" min="0.1" options={capabilities?.durations ?? []} step="0.0001" onChange={(value) => mutateShot((next) => { next.targetDurationSeconds = Number(value); })} value={shot.targetDurationSeconds} />
              <label>Resolution override<input value={shot.resolution ?? ""} onChange={(event) => mutateShot((next) => setOptional(next, "resolution", event.target.value))} /></label>
              <label>Start state<textarea value={shot.startState} onChange={(event) => mutateShot((next) => { next.startState = event.target.value; })} /></label>
              <label>End state<textarea value={shot.endState} onChange={(event) => mutateShot((next) => { next.endState = event.target.value; })} /></label>
              <NumberInput label="Seed" onChange={(value) => mutateShot((next) => setOptionalNumber(next, "seed", value))} value={shot.seed} />
              <Finding field="" findings={findings} shotId={shot.id} />
            </fieldset>
            <fieldset className="ve-film-inspector-fields" disabled={disabled}>
              <legend>Conditioning and continuity</legend>
              <label>Mode<select aria-label="Conditioning mode" value={shot.conditioning.mode} onChange={(event) => mutateShot((next) => { next.conditioning.mode = event.target.value; })}>{modes.map((mode) => <option key={mode} value={mode}>{mode}</option>)}</select></label>
              <label>First frame role<select value={shot.conditioning.firstFrameRole ?? ""} onChange={(event) => mutateShot((next) => setOptional(next.conditioning, "firstFrameRole", event.target.value))}><option value="">None</option>{conditionableRoles.map((role) => <option key={role}>{role}</option>)}</select></label>
              <label>Last frame role<select value={shot.conditioning.lastFrameRole ?? ""} onChange={(event) => mutateShot((next) => setOptional(next.conditioning, "lastFrameRole", event.target.value))}><option value="">None</option>{conditionableRoles.map((role) => <option key={role}>{role}</option>)}</select></label>
              <label>Ordered reference roles<input aria-label="Reference roles" list="film-reference-role-options" value={(shot.conditioning.referenceRoles ?? []).join(", ")} onChange={(event) => mutateShot((next) => { next.conditioning.referenceRoles = asList(event.target.value); })} /></label>
              <datalist id="film-reference-role-options">{conditionableRoles.map((role) => <option key={role} value={role} />)}</datalist>
              <label>Continuity chain from<select value={shot.conditioning.chainFromShotId ?? ""} onChange={(event) => mutateShot((next) => setOptional(next.conditioning, "chainFromShotId", event.target.value))}><option value="">None</option>{shots.filter((item) => item.id !== shot.id).map((item) => <option key={item.id}>{item.id}</option>)}</select></label>
              <label>Continuity roles<input aria-label="Continuity roles" list="film-continuity-role-options" value={(shot.continuityRoles ?? []).join(", ")} onChange={(event) => mutateShot((next) => { next.continuityRoles = asList(event.target.value); })} /></label>
              <datalist id="film-continuity-role-options">{approvedRoles.map((role) => <option key={role} value={role} />)}</datalist>
              <Finding field="conditioning" findings={findings} shotId={shot.id} /><Finding field="continuityRoles" findings={findings} shotId={shot.id} />
              <div className="ve-film-dependencies"><strong>Dependencies</strong>{(shot.dependsOn ?? []).map((dependency, index) => <div className="ve-film-dependency" key={`${dependency.shotId}-${index}`}><select aria-label={`Dependency ${index + 1} shot`} value={dependency.shotId} onChange={(event) => mutateShot((next) => { next.dependsOn[index].shotId = event.target.value; })}><option value="">Choose shot</option>{shots.filter((item) => item.id !== shot.id).map((item) => <option key={item.id}>{item.id}</option>)}</select><select aria-label={`Dependency ${index + 1} kind`} value={dependency.kind} onChange={(event) => mutateShot((next) => { next.dependsOn[index].kind = event.target.value; })}>{DEPENDENCY_KINDS.map((kind) => <option key={kind}>{kind}</option>)}</select><input aria-label={`Dependency ${index + 1} note`} placeholder="Intent note" value={dependency.note ?? ""} onChange={(event) => mutateShot((next) => setOptional(next.dependsOn[index], "note", event.target.value))} /><button onClick={() => mutateShot((next) => { next.dependsOn.splice(index, 1); })} type="button">Remove</button></div>)}<button onClick={() => mutateShot((next) => { next.dependsOn ??= []; next.dependsOn.push({ shotId: "", kind: "continuity" }); })} type="button">Add dependency</button></div>
              <Finding field="dependsOn" findings={findings} shotId={shot.id} />
            </fieldset>
            <fieldset className="ve-film-inspector-fields" disabled={disabled}>
              <legend>Dialogue placement</legend>
              <label>Audio<textarea aria-label={`Shot ${shot.id} audio`} placeholder="What this shot sounds like, or that it is silent" rows="2" value={shot.audio ?? ""} onChange={(event) => mutateShot((next) => { next.audio = event.target.value; })} /></label>
              <Finding field="audio" findings={findings} shotId={shot.id} />
              <label>Dialogue<textarea aria-label={`Shot ${shot.id} dialogue`} rows="2" value={shot.dialogue ?? ""} onChange={(event) => mutateShot((next) => setOptional(next, "dialogue", event.target.value))} /></label>
              <label>Generated picture audio<select aria-label={`Shot ${shot.id} generated audio`} value={shot.generatedAudio ?? ""} onChange={(event) => mutateShot((next) => setOptional(next, "generatedAudio", event.target.value))}><option value="">Use film default</option><option value="mute">Mute</option><option value="include">Include</option></select></label>
              <label>Audio role<input value={shot.dialogueClip?.role ?? ""} onChange={(event) => mutateShot((next) => { next.dialogueClip ??= { role: "", offsetSeconds: 0, gain: 1, sourceInSeconds: 0 }; next.dialogueClip.role = event.target.value; })} /></label>
              {[['offsetSeconds', 'Timeline offset'], ['sourceInSeconds', 'Source in'], ['durationSeconds', 'Duration'], ['gain', 'Gain'], ['fadeInSeconds', 'Fade in'], ['fadeOutSeconds', 'Fade out']].map(([key, label]) => <NumberInput key={key} label={label} min={key === "gain" ? undefined : "0"} step="0.01" value={shot.dialogueClip?.[key]} onChange={(value) => mutateShot((next) => { next.dialogueClip ??= { role: "", offsetSeconds: 0, gain: 1, sourceInSeconds: 0 }; setOptionalNumber(next.dialogueClip, key, value); })} />)}
              {shot.dialogueClip ? <button onClick={() => mutateShot((next) => { delete next.dialogueClip; })} type="button">Clear placement</button> : null}
              <Finding field="dialogue" findings={findings} shotId={shot.id} /><Finding field="dialogueClip" findings={findings} shotId={shot.id} />
            </fieldset>
          </article>
        ) : null}
      </div>

      {compiled ? <section className="ve-film-preflight" aria-label="Render preflight"><h4>Effective requests</h4>{compiled.requests.filter((request) => selectedShotIds.includes(request.shotId)).map((request) => <article key={request.shotId}><strong>{request.shotId}</strong><dl><dt>Model</dt><dd>{request.model}{request.partitionReason ? ` · ${request.partitionReason}` : ""}</dd><dt>Output</dt><dd>{request.width}×{request.height} · {request.fps} fps · {request.durationSeconds}s</dd><dt>Conditioning</dt><dd>{request.mode} · {request.referenceRoles?.join(", ") || "no reference roles"} · reference edge {request.referenceImageShortEdge ?? "model default"}</dd><dt>Sampling</dt><dd>{request.effectiveSteps ?? "model default"} steps · {request.loras?.join(", ") || "base model"} · seed {request.seed ?? "random"}</dd><dt>Prompt</dt><dd>{request.promptSource}{request.promptSource === "refined" ? " (recorded planner identity below)" : ""}</dd></dl></article>)}{compiled.planner?.executions?.length ? <div><strong>Prompt refinement identity</strong><ul>{compiled.planner.executions.map((execution, index) => <li key={`${execution.jobId ?? "execution"}-${index}`}>{execution.provider} · {execution.model} · thinking {execution.thinkingMode} · target {execution.targetVideoModelId}</li>)}</ul></div> : <p>Authored prompts; no prompt-refinement model was invoked.</p>}</section> : null}
    </section>
  );
}
