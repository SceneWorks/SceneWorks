import { Segmented } from "./LabFields.jsx";
import React from "react";
import { Icon } from "../../components/Icons.jsx";
import { ErrorNotice, ScorePreview } from "../audioYue2Parts.jsx";

export function Yue2ComposePanel({
  update,
  settings,
  sessionOnlyNote,
  inspect,
  preview,
  restorable,
  submitting,
  composeProblems,
  readyToRun,
  submit,
  kind,
  planWords,
  tierProblem,
  composeLabel
}) {
  return (
<div className="yue2-panel" data-testid="yue2-compose">
                <div className="yue2-inline">
                  <span className="eyebrow">Plan</span>
                  <Segmented
                    label="Plan source"
                    onChange={(value) => update({ planSource: value })}
                    options={[
                      { value: "sample", label: "Sample a new plan" },
                      { value: "supplied", label: "Supply an ABC score" },
                      { value: "restore", label: "Restore a saved plan" },
                    ]}
                    value={settings.planSource}
                  />
                </div>
                {settings.planSource !== "restore" ? (
                  <div className="yue2-inline">
                    <span className="eyebrow">Planning</span>
                    <Segmented
                      disabledValues={settings.planSource === "supplied" || settings.planOnly ? ["off"] : []}
                      label="Planning"
                      onChange={(value) => update({ planning: value })}
                      options={[
                        { value: "full", label: "Full (melody + harmony)" },
                        { value: "melody", label: "Melody only" },
                        { value: "off", label: "Off (no score)" },
                      ]}
                      value={settings.planning}
                    />
                    <label className="checkline">
                      <input
                        checked={settings.planOnly}
                        onChange={(event) => update({ planOnly: event.target.checked })}
                        type="checkbox"
                      />
                      Plan only (stop after the score)
                    </label>
                  </div>
                ) : null}
                {settings.planSource === "supplied" ? (
                  <div className="yue2-abc-block">
                    <textarea
                      aria-label="Supplied ABC score"
                      className="yue2-abc"
                      onChange={(event) => update({ suppliedScore: event.target.value })}
                      placeholder="X:1 … two voices (Vocal / Ins) in the native YuE2 dialect"
                      rows={8}
                      spellCheck={false}
                      value={settings.suppliedScore}
                    />
                    {sessionOnlyNote(settings.suppliedScore)}
                    <button
                      className="secondary-action"
                      disabled={!settings.suppliedScore.trim()}
                      onClick={() => inspect("supplied", settings.suppliedScore)}
                      type="button"
                    >
                      Preview score
                    </button>
                    {preview.key === "supplied" ? (
                      <>
                        <ErrorNotice error={preview.error} testId="yue2-preview-error" />
                        <ScorePreview inspection={preview.inspection} />
                      </>
                    ) : null}
                  </div>
                ) : null}
                {settings.planSource === "restore" ? (
                  <label>
                    Saved plan
                    <select
                      aria-label="Saved plan"
                      onChange={(event) => update({ restorePlanJobId: event.target.value })}
                      value={settings.restorePlanJobId}
                    >
                      <option value="">Choose a finished plan or song…</option>
                      {restorable.map((view) => (
                        <option key={view.id} value={view.id}>
                          {view.kindLabel} · {view.style || view.id}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                <div className="prompt-cta-stack yue2-cta">
                  <button
                    className="prompt-cta"
                    disabled={submitting || composeProblems.length > 0 || !readyToRun}
                    onClick={() => submit(kind, kind === "fromPlan" ? { plan: planWords } : {})}
                    title={composeProblems.join(" ") || tierProblem || undefined}
                    type="button"
                  >
                    <Icon.Sparkle size={14} /> {composeLabel}
                  </button>
                  {composeProblems.length ? <span className="yue2-muted">{composeProblems.join(" ")}</span> : null}
                </div>
              </div>
  );
}
