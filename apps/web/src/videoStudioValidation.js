// The Video Studio Generate gate in the app-wide vocabulary (epic 10644, sc-10650).
//
// This screen carried the epic's cleanest drift bug: `canSubmit` computed readiness from
// nine conditions, and a separate `blockedMessage` ternary re-derived a human-readable
// reason for five of them — two parallel expressions of the same rules, synced by hand.
// Here the reason and the gate are the same issue, so one cannot say "ready" while the
// other says why it isn't.

import { presetLoraIssues } from "./generationValidation.js";
import { promptBudget } from "./styleComposer.js";
import { issue } from "./validation/issues.js";

// SCAIL-2's catalog id, named once so the panel that HIDES the Replacement mode control and the
// studio that stops SENDING it key on one spelling rather than two literals that can drift.
export const SCAIL2_MODEL_ID = "scail2_14b";

// Whether a replacement engine actually consumes `replacementMode`. Today it is a single negative
// fact rather than a per-model table: SCAIL-2 re-renders the whole tracked person from the character
// reference, so face-only / keep-outfit has nothing to select, and every scail2 conditioning site in
// the worker emits `ReplacementMode::default()` LITERALLY — the user's choice never reached the
// engine. The engine now refuses a non-default mode outright (sc-20262), so a control that can still
// set one is a refusal waiting to happen rather than a knob. The Wan-VACE inpainting engines DO
// honor it and are unchanged.
//
// The studio and ReplacePersonPanel share this predicate so the visible control and submitted
// request agree. The panel imports this module, which keeps the dependency direction acyclic.
//
// sc-20262 is the story that would make the mode honored. If it lands, delete this predicate rather
// than adding a second model to it.
export function replacementModeApplies(modelId) {
  return modelId !== SCAIL2_MODEL_ID;
}

export function videoGenerateValidation({
  activeProject,
  promptless,
  prompt,
  implementedMode,
  hasInputs,
  replaceReady,
  // A backend/model execution tier can be intentionally narrower than Model Manager's installable
  // package set. The studio derives the actionable reason from the exact active lane and passes it
  // here so the visible summary and Generate gate remain one authority.
  executionTierBlockMessage = null,
  vedaBlockMessage = null,
  modelName,
  // sc-13136: the COMPOSED outgoing prompt (Subject:/Style: wrap + preset fold) and whether a
  // Style Catalog entry is active. `composedPrompt` is the exact string that will be sent — the same
  // string the live preview shows — so the cap is measured on IT, not the raw prompt field: a
  // ~700–900 char style wrapped around a long-but-under-cap prompt can compose past the backend cap.
  styleActive = false,
  composedPrompt = "",
  presetMissing = [],
  presetIncompatible = [],
  loraIncompatible = [],
} = {}) {
  const issues = [];
  if (!activeProject) {
    issues.push(issue.requirement("project", "Open a project to generate"));
  }
  // Image-conditioned models take no prompt; only gate on prompt text when one is expected.
  if (!promptless && !prompt?.trim()) {
    issues.push(issue.requirement("prompt", "Write a prompt"));
  }
  // Composed-prompt budget guard (sc-13136, mirrors image sc-13133). ONLY when a style is active:
  // styleless behavior is unchanged. An error, not a silent requirement — nothing else on the form
  // explains why Generate is dead, and we warn rather than let the run reach the backend's reject.
  if (styleActive) {
    const budget = promptBudget(composedPrompt);
    if (budget.over) {
      issues.push(
        issue.error(
          null,
          `Prompt with this style is ${budget.length}/${budget.max} characters — shorten your prompt or pick a shorter style.`,
        ),
      );
    }
  }
  // The mode's inputs (source clip, reference images) are visible upload zones — an empty
  // one speaks for itself, so this is a silent requirement. It drops the old vague
  // "Required inputs are missing" message, which never named what was missing anyway.
  if (!hasInputs) {
    issues.push(issue.requirement("inputs", "Add the inputs this mode needs"));
  }
  // An entry point without a runtime still needs a product-level notice.
  if (!implementedMode) {
    issues.push(issue.error(null, "This entry point is reserved for the next runtime slice."));
  }
  if (!replaceReady) {
    issues.push(issue.error(null, "No live GPU worker can run person replacement yet."));
  }
  if (vedaBlockMessage) {
    issues.push(issue.error(null, vedaBlockMessage));
  }
  if (executionTierBlockMessage) {
    issues.push(issue.error(null, executionTierBlockMessage));
  }
  issues.push(...presetLoraIssues({ presetMissing, presetIncompatible, loraIncompatible, modelName }));
  return issues;
}
