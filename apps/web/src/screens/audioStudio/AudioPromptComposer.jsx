import { MODE_PLACEHOLDER } from "./audioStudioHelpers.js";
import React from "react";
import { Icon } from "../../components/Icons.jsx";
import { RefinePromptControl } from "../../components/RefinePromptControl.jsx";
import { YueLyricsEditor } from "../audioYueControls.jsx";
import { splitGenreTagLine } from "../../yueSong.js";

export function AudioPromptComposer({
  songModel,
  setYueSections,
  yueSections,
  showMultiSpeaker,
  script,
  updateSegment,
  speakerChoices,
  mode,
  selectedModel,
  promptGuide,
  refinePrompt,
  refineModel,
  createModelDownloadJob,
  removeSegment,
  addSegment,
  setPrompt,
  prompt,
  canGenerate,
  setGenreTags,
  songPrompt
}) {
  return (
<>
<div className="prompt-input-row">
              {/* Multi-speaker / long-form dialogue (sc-13676): the plain prompt is replaced by a
                  segmented-script editor ONLY when the selected model advertises
                  audio.supportsMultiSpeaker. Each turn carries a speaker (capped at maxSpeakers, read
                  off the model) + its text; the script submits as AudioParams.script. Capability-driven
                  — every single-voice Speech model keeps the plain textarea, so those modes are
                  unperturbed. */}
              {/* YuE (sc-19385): a segmented-song model replaces the prompt with its section-labelled
                  lyrics editor; its prompt is the genre tags in the settings bar below. */}
              {songModel ? (
                <YueLyricsEditor onChange={setYueSections} sections={yueSections} />
              ) : showMultiSpeaker ? (
                <div
                  className="prompt-input multi-speaker-script"
                  data-testid="multi-speaker-script"
                  role="group"
                  aria-label="Multi-speaker script"
                >
                  {script.map((segment, index) => (
                    <div className="script-segment" key={index}>
                      <select
                        aria-label={`Segment ${index + 1} speaker`}
                        className="script-segment-speaker"
                        onChange={(event) => updateSegment(index, { speaker: event.target.value })}
                        value={segment.speaker ?? speakerChoices[0]?.value ?? "S1"}
                      >
                        {speakerChoices.map((choice) => (
                          <option key={choice.value} value={choice.value}>
                            {choice.label}
                          </option>
                        ))}
                      </select>
                      <div className="script-segment-content">
                        <textarea
                          aria-label={`Segment ${index + 1} text`}
                          className="script-segment-text"
                          onChange={(event) => updateSegment(index, { text: event.target.value })}
                          placeholder="What this speaker says…"
                          rows={2}
                          value={segment.text ?? ""}
                        />
                        <RefinePromptControl
                          key={`${mode}:${selectedModel?.id}:${script.length}:${index}`}
                          guidePath={promptGuide.path}
                          modelId={selectedModel?.id}
                          onApply={(refined) => updateSegment(index, { text: refined })}
                          prompt={segment.text}
                          refinePrompt={refinePrompt}
                          refineModel={refineModel}
                          onDownloadRefineModel={
                            refineModel ? () => createModelDownloadJob(refineModel) : undefined
                          }
                          workflow="audio"
                        />
                      </div>
                      <button
                        aria-label={`Remove segment ${index + 1}`}
                        className="script-segment-remove"
                        disabled={script.length <= 1}
                        onClick={() => removeSegment(index)}
                        title="Remove this turn"
                        type="button"
                      >
                        <Icon.Trash size={14} />
                      </button>
                    </div>
                  ))}
                  <button
                    className="script-add-segment"
                    data-testid="script-add-segment"
                    onClick={addSegment}
                    type="button"
                  >
                    <Icon.Plus size={14} />
                    Add turn
                  </button>
                </div>
              ) : (
                <textarea
                  aria-label="Prompt"
                  className="prompt-input"
                  onChange={(event) => setPrompt(event.target.value)}
                  placeholder={MODE_PLACEHOLDER[mode] ?? "Describe the audio…"}
                  value={prompt}
                />
              )}
              {/* The shell's primary action. C1 (sc-13408) wires the real Speech submission via the
                  form's onSubmit. Disabled — never a silent no-op — until a run can proceed: empty
                  content (prompt or multi-speaker script), no installed model, a non-Speech tab, or a
                  run already in flight all block it. */}
              {/* The CTA and the shortcut hint that used to live in the results head — the hint
                  belongs with the action it describes (epic 14361, screen 2a). */}
              <div className="prompt-cta-stack">
                <button className="prompt-cta" type="submit" disabled={!canGenerate}>
                  <Icon.Sparkle size={14} />
                  Generate
                </button>
                <span className="kbd-hint">
                  <kbd>⌘</kbd>
                  <kbd>↵</kbd>
                  to generate
                </span>
              </div>
            </div>

            {/* Multi-speaker turns own one refiner apiece inside their structured editor above. Plain
                prompt modes share this single control. Key by mode + resolved model so switching either
                aborts an in-flight request and cannot surface a stale rewrite from the previous guide. */}
            {/* A segmented-song model's prompt is its genre-tag list, so the refiner reads and writes
                that (a refined tag line is re-split into tags — commas, else known multi-word tags, else words). */}
            {!showMultiSpeaker && selectedModel ? (
              <RefinePromptControl
                key={`${mode}:${selectedModel.id}`}
                guidePath={promptGuide.path}
                modelId={selectedModel.id}
                onApply={songModel ? (refined) => setGenreTags(splitGenreTagLine(refined)) : setPrompt}
                prompt={songModel ? songPrompt : prompt}
                refinePrompt={refinePrompt}
                refineModel={refineModel}
                onDownloadRefineModel={
                  refineModel ? () => createModelDownloadJob(refineModel) : undefined
                }
                workflow="audio"
              />
            ) : null}
</>
  );
}
