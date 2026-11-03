import { MOTIONS } from "./videoTierHelpers.js";
import React from "react";
import { Icon } from "../../components/Icons.jsx";
import { RefinePromptControl } from "../../components/RefinePromptControl.jsx";
import { onPromptKeyDown, ModeTabs } from "../../components/generationStudio.jsx";

export function VideoPromptComposer({
  modeOptions,
  mode,
  setMode,
  modeTabBlocked,
  modeTabBlockedText,
  setGuideOpen,
  onOpenPresets,
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
  createModelDownloadJob,
  motion,
  setMotion
}) {
  return (
<>
<div className="prompt-hero-top">
            <ModeTabs
              className="mode-tabs mode-control"
              label="Video mode"
              options={modeOptions}
              mode={mode}
              onChange={setMode}
              blockFor={(value, active) => !active && modeTabBlocked(value)
                ? { text: modeTabBlockedText }
                : null}
            />
            <div className="prompt-hero-links">
              <button className="hero-link" onClick={() => setGuideOpen(true)} type="button">
                <Icon.Book size={14} /> Prompt guide
              </button>
              {onOpenPresets ? (
                <button className="hero-link" onClick={onOpenPresets} type="button">
                  <Icon.Folder size={14} /> Saved presets
                </button>
              ) : null}
            </div>
          </div>

          <div className="prompt-input-row">
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
            <button className="prompt-cta" disabled={generateDisabled} type="submit">
              <Icon.Sparkle size={14} />
              {submitting ? "Queueing…" : renderLabel}
            </button>
          </div>

          {promptless ? null : (
            <RefinePromptControl
              guidePath={promptGuide.path}
              modelId={model}
              onApply={setPrompt}
              prompt={prompt}
              refinePrompt={refinePrompt}
              refineModel={refineModel}
              onDownloadRefineModel={refineModel ? () => createModelDownloadJob(refineModel) : undefined}
              workflow="video"
            />
          )}

          {/* Style Catalog picker + its composed-prompt preview moved into the Style axis row of the
              settings bar (sc-13135) — see the .settings-bar-style-axis block below. */}

          <div className="motion-row">
            <span className="motion-row-label">Motion:</span>
            {MOTIONS.map((option) => (
              <button
                className={motion === option ? "motion-chip active" : "motion-chip"}
                key={option}
                onClick={() => setMotion(option)}
                type="button"
              >
                <span aria-hidden="true" className="motion-arrow">→</span>
                {option}
              </button>
            ))}
          </div>
</>
  );
}
