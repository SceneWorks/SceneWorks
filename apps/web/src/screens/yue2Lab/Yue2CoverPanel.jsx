import { Segmented } from "./LabFields.jsx";
import React from "react";
import { Icon } from "../../components/Icons.jsx";
import { ErrorNotice, RegenerationNotice, ScorePreview } from "../audioYue2Parts.jsx";
import { Yue2RecordingCover } from "../audioYue2Recording.jsx";

export function Yue2CoverPanel({
  assets,
  importAsset,
  jobs,
  model,
  jobAction,
  licenseLapsed,
  submit,
  projectId,
  finishedKey,
  requestedGpu,
  runs,
  settings,
  submitting,
  token,
  update,
  versions,
  inspectCoverVersion,
  sessionOnlyNote,
  inspect,
  preview,
  regenerationNotice,
  coverVersion,
  reviewedTranscriptionVersionId,
  setReviewedTranscriptionVersionId,
  coverProblems,
  readyToRun,
  tierProblem
}) {
  return (
<div className="yue2-panel" data-testid="yue2-cover">
                <Yue2RecordingCover
                  assets={assets}
                  importAsset={importAsset}
                  jobs={jobs}
                  model={model}
                  onCancel={(target) => jobAction?.(target, "cancel")}
                  onLicenseLapsed={licenseLapsed}
                  onTranscribe={(target) => submit("transcribe", target)}
                  projectId={projectId}
                  refreshKey={finishedKey}
                  requestedGpu={requestedGpu}
                  runs={runs}
                  settings={settings}
                  submitting={submitting}
                  token={token}
                  update={update}
                  versions={versions}
                />
                <div className="yue2-inline">
                  <span className="eyebrow">Cover score</span>
                  <Segmented
                    label="Cover score source"
                    onChange={(value) => update({ coverSource: value })}
                    options={[
                      { value: "version", label: "Score version" },
                      { value: "inline", label: "Paste ABC" },
                    ]}
                    value={settings.coverSource}
                  />
                </div>
                {settings.coverSource === "version" ? (
                  <div className="yue2-inline">
                    <select
                      aria-label="Cover score version"
                      onChange={(event) => update({ coverVersionId: event.target.value })}
                      value={settings.coverVersionId}
                    >
                      <option value="">Choose a reviewed score version…</option>
                      {versions.map((version) => (
                        <option key={version.id} value={version.id}>
                          {version.id} · {version.editOperation ?? version.origin} · {version.cot}
                          {version.transcription ? ` · from a recording (${version.transcription.mode})` : ""}
                          {version.nonCommercial ? " · Noncommercial" : ""}
                        </option>
                      ))}
                    </select>
                    <button
                      className="secondary-action"
                      disabled={!settings.coverVersionId}
                      onClick={() => inspectCoverVersion(settings.coverVersionId)}
                      type="button"
                    >
                      Review score
                    </button>
                    <button
                      className="secondary-action"
                      disabled={!settings.coverVersionId}
                      onClick={() => update({ tab: "scores", selectedVersionId: settings.coverVersionId })}
                      type="button"
                    >
                      Edit in Scores
                    </button>
                  </div>
                ) : (
                  <div className="yue2-abc-block">
                    <textarea
                      aria-label="Cover ABC score"
                      className="yue2-abc"
                      onChange={(event) => update({ coverScore: event.target.value })}
                      rows={8}
                      spellCheck={false}
                      value={settings.coverScore}
                    />
                    {sessionOnlyNote(settings.coverScore)}
                    <button
                      className="secondary-action"
                      disabled={!settings.coverScore.trim()}
                      onClick={() => inspect("coverInline", settings.coverScore)}
                      type="button"
                    >
                      Review score
                    </button>
                  </div>
                )}
                {preview.key === "coverVersion" || preview.key === "coverInline" ? (
                  <div data-testid="yue2-cover-review">
                    <ErrorNotice error={preview.error} testId="yue2-preview-error" />
                    <ScorePreview inspection={preview.inspection} />
                  </div>
                ) : null}
                <div className="yue2-grid">
                  <label>
                    Cover mode
                    <select aria-label="Cover mode" onChange={(event) => update({ coverMode: event.target.value })} value={settings.coverMode}>
                      <option value="melody">Melody</option>
                      <option value="full">Full</option>
                    </select>
                  </label>
                  <label title={settings.coverMode === "melody" ? undefined : "Only a melody cover chooses which melodies it keeps."}>
                    Keep
                    <select aria-label="Cover keep" disabled={settings.coverMode !== "melody"} onChange={(event) => update({ coverKeep: event.target.value })} value={settings.coverKeep}>
                      <option value="">Model default</option>
                      <option value="both">Vocal and instrumental</option>
                      <option value="vocal">Vocal only</option>
                      <option value="instrumental">Instrumental only</option>
                    </select>
                  </label>
                </div>
                <label>
                  Source lyrics (when the lyrics above are their translation)
                  <textarea
                    aria-label="Translated from"
                    onChange={(event) => update({ coverTranslatedFrom: event.target.value })}
                    rows={3}
                    value={settings.coverTranslatedFrom}
                  />
                </label>
                <RegenerationNotice text={regenerationNotice} />
                {settings.coverSource === "version" && coverVersion?.transcription ? (
                  <label className="model-license-ack" data-testid="yue2-transcription-review-confirmation">
                    <input
                      checked={reviewedTranscriptionVersionId === settings.coverVersionId}
                      onChange={(event) => setReviewedTranscriptionVersionId(event.target.checked ? settings.coverVersionId : "")}
                      type="checkbox"
                    />
                    I reviewed this transcription, its warnings and octave evidence, and this score version
                  </label>
                ) : null}
                <div className="prompt-cta-stack yue2-cta">
                  <button
                    className="prompt-cta"
                    disabled={submitting || coverProblems.length > 0 || !readyToRun}
                    onClick={() => submit("cover")}
                    title={coverProblems.join(" ") || tierProblem || undefined}
                    type="button"
                  >
                    <Icon.Sparkle size={14} /> Generate cover
                  </button>
                  {coverProblems.length ? <span className="yue2-muted">{coverProblems.join(" ")}</span> : null}
                </div>
              </div>
  );
}
