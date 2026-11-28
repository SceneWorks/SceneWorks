import { MOTIONS } from "./videoTierHelpers.js";
import React, { useState } from "react";
import { Icon } from "../../components/Icons.jsx";
import { RefinePromptControl } from "../../components/RefinePromptControl.jsx";
import { onPromptKeyDown } from "../../components/generationStudio.jsx";
import { SUBMIT_SHORTCUT_ARIA, SUBMIT_SHORTCUT_LABEL } from "../../components/submitShortcut.js";

const motionLabel = (value) => value.charAt(0).toUpperCase() + value.slice(1);

// The prompt box (studio layout simplification): the textarea, then a footer bar with the prompt
// tools on the left (Refine, Motion) and the submit action on the right. `status` — Generate
// errors, portable-option status, the Veda receipt, Start new generation — renders directly under
// the box; the Refine panel opens as one inline card beneath that.
export function VideoPromptComposer({
  setPrompt,
  promptless,
  prompt,
  generateDisabled,
  submitting,
  renderLabel,
  promptGuide,
  model,
  refinePrompt,
  refineModel,
  loopModel,
  createModelDownloadJob,
  motion,
  setMotion,
  status = null,
}) {
  const [refineOpen, setRefineOpen] = useState(false);
  return (
    <>
      <div className="studio-prompt-box">
        <textarea
          aria-label="Prompt"
          className="prompt-input"
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={onPromptKeyDown}
          placeholder={
            promptless
              ? "No prompt needed — this model animates the source image. Pick a first frame below and generate."
              : "Describe the motion — what moves, where the camera goes, how it feels…"
          }
          value={prompt}
        />
        <div className="studio-prompt-box-foot">
          {promptless ? null : (
            <>
              <button
                aria-controls="video-refine-card"
                aria-pressed={refineOpen}
                className="studio-prompt-tool"
                onClick={() => setRefineOpen((open) => !open)}
                type="button"
              >
                <Icon.Wand size={16} />
                Refine
              </button>
              <span aria-hidden="true" className="studio-prompt-box-divider" />
            </>
          )}
          {/* Motion writes the same `motion` state the chip row did; the submit folds it into the
              request exactly as before. */}
          <label className="video-motion-field">
            Motion
            <select aria-label="Motion" onChange={(event) => setMotion(event.target.value)} value={motion}>
              {MOTIONS.map((option) => (
                <option key={option} value={option}>
                  {motionLabel(option)}
                </option>
              ))}
            </select>
          </label>
          <span aria-hidden="true" className="studio-prompt-box-spacer" />
          <button
            aria-keyshortcuts={SUBMIT_SHORTCUT_ARIA}
            className="studio-generate"
            disabled={generateDisabled}
            type="submit"
          >
            <Icon.Sparkle size={16} />
            {submitting ? "Queueing…" : renderLabel}
            <span className="kbd" aria-hidden="true">{SUBMIT_SHORTCUT_LABEL}</span>
          </button>
        </div>
      </div>

      {status}

      {/* The Refine panel stays MOUNTED while closed (hidden), so an in-flight refinement or a
          pending Apply / Keep original review survives closing and reopening the panel. */}
      {promptless ? null : (
        <div className="studio-tool-card video-refine-card" hidden={!refineOpen} id="video-refine-card">
          <div className="studio-tool-card-head">
            <span className="studio-tool-card-title">Refine</span>
            <span className="studio-tool-card-hint">Rewrite what you typed for clarity and detail</span>
            <span className="studio-prompt-box-spacer" />
            <button
              aria-label="Close Refine panel"
              className="video-tool-card-close"
              onClick={() => setRefineOpen(false)}
              type="button"
            >
              <Icon.Close size={16} />
            </button>
          </div>
          <RefinePromptControl
            guidePath={promptGuide.path}
            modelId={model}
            onApply={setPrompt}
            prompt={prompt}
            refinePrompt={refinePrompt}
            refineModel={refineModel}
            loopModel={loopModel}
            onDownloadLoopModel={createModelDownloadJob}
            onDownloadRefineModel={refineModel ? () => createModelDownloadJob(refineModel) : undefined}
            workflow="video"
          />
        </div>
      )}
    </>
  );
}
