import React from "react";

import { promptBudget } from "../styleComposer.js";

// sc-13131 — Live composed-prompt preview for the selected Style Catalog entry.
//
// Shows the EXACT outgoing `prompt` string the run will send once a style is active — the
// `Subject:`/`Style:` composition, any preserved sibling directive lines, and (when the user
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
// Subject/Style line breaks stay legible (styles.css `.styled-prompt-preview`).
// sc-13133: a style wraps ~700–900 chars around the user's prompt, and the backend rejects a
// composed prompt over the cap (see promptBudget / PROMPT_MAX_CHARS). Because a style is active
// exactly when this preview renders, the composed length / remaining budget belongs here: a live
// "N / 4000" readout beside the string it measures, flipping to an over-budget warning before the
// user submits. The measurement is on the SAME composed string shown above, so the count and the
// blocking Generate error (imageGenerateValidation) can never disagree.
export function StyledPromptPreview({ active, composedPrompt }) {
  if (!active) {
    return null;
  }
  const budget = promptBudget(composedPrompt ?? "");
  return (
    <div className="preset-stack-preview styled-prompt-preview" data-testid="styled-prompt-preview">
      <div className="preset-stack-prompt">
        <span className="eyebrow">Style-composed prompt sent</span>
        <p>{composedPrompt ? composedPrompt : <span className="token">your prompt</span>}</p>
      </div>
      <p
        className={budget.over ? "styled-prompt-budget over" : "styled-prompt-budget"}
        role={budget.over ? "alert" : undefined}
        data-testid="styled-prompt-budget"
      >
        {budget.over ? (
          <>
            Too long: {budget.length} / {budget.max} characters — shorten your prompt or pick a shorter style.
          </>
        ) : (
          <>
            {budget.length} / {budget.max} characters
          </>
        )}
      </p>
    </div>
  );
}
