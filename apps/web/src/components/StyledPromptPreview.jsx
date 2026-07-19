import React from "react";

// sc-13131 — Live composed-prompt preview for the selected Style Catalog entry.
//
// Shows the EXACT outgoing `prompt` string the run will send once a style is active — the
// `Style:`/`Description:` composition, any preserved sibling directive lines, and (when the user
// typed their own `Style:` line) the MERGE — so the splice is never silent (R4). It updates live
// as the user types, swaps the selected style, or changes presets because the caller recomputes
// `composedPrompt` on every render from the SAME `buildJobRequest` the single Generate submit
// uses; the string shown here therefore cannot drift from what is submitted.
//
// Renders nothing when no style is active (`active` false) — an inactive style has nothing extra
// to preview, so we hide the affordance rather than echo the plain prompt misleadingly.
//
// Visual idiom mirrors PresetStackPreview (generationStudio.jsx): the same framed
// `preset-stack-preview` container + `preset-stack-prompt` monospace block, with `pre-wrap` so the
// Style/Description line breaks stay legible (styles.css `.styled-prompt-preview`).
export function StyledPromptPreview({ active, composedPrompt }) {
  if (!active) {
    return null;
  }
  return (
    <div className="preset-stack-preview styled-prompt-preview" data-testid="styled-prompt-preview">
      <div className="preset-stack-prompt">
        <span className="eyebrow">Style-composed prompt sent</span>
        <p>{composedPrompt ? composedPrompt : <span className="token">your prompt</span>}</p>
      </div>
    </div>
  );
}
