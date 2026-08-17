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
// It lives HERE, rather than beside the control in `ReplacePersonPanel.jsx`, for two reasons that
// both point the same way. `ReplacePersonPanel.jsx` imports this module and not the reverse, so
// there is no cycle; and this file is a fingerprinted source of `config/backend-capabilities/
// matrix.json` (`webVideoValidation`) while the panel is not — a predicate that decides what the
// studio SENDS must sit inside that fingerprint, or it could be edited without the derived matrix
// noticing. `VideoStudio.jsx` is fingerprinted too, so both readers are covered.
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
  supportsMode,
  implementedMode,
  hasInputs,
  requiresLtxIcLora,
  hasLtxIcLora,
  replaceReady,
  // sc-17161: the refusal from `referenceLimitError` for a reference selection past what the model
  // declares (`limits.max{Reference,SourceClip,ReferenceAudio,CombinedReference}Assets`), or null.
  // An ERROR, not a silent requirement: the pickers all look satisfied — MiniMax-H3 Ref2VA's
  // combined cap refuses selections in which every individual picker is inside its own cap — so
  // nothing else on the form explains why Generate is dead.
  referenceLimitMessage = null,
  // sc-19574: `reference_to_video` with audio references selected and NO image or video reference.
  // An ERROR rather than the silent `inputs` requirement, for `referenceLimitMessage`'s reason: the
  // audio picker is visibly full, so an empty image zone next to it reads as optional. The rule is
  // the reference implementation's own — diffusers `MiniMaxH3` raises on `set(kinds) == {"audio"}`,
  // because an audio reference never reaches the visual conditioner — and both the API and the
  // worker refuse the same shape, so saying it here is where the user finds out first.
  audioOnlyReferenceSet = false,
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
  // A mode the model can't run, or a runtime entry point not built yet. Nothing on the
  // form explains either, so they speak.
  if (!supportsMode) {
    issues.push(issue.error(null, `${modelName ?? "Selected model"} does not support this mode.`));
  }
  if (!implementedMode) {
    issues.push(issue.error(null, "This entry point is reserved for the next runtime slice."));
  }
  if (requiresLtxIcLora && !hasLtxIcLora) {
    issues.push(issue.error(null, "LTX video-conditioned generation needs a selected IC-LoRA adapter."));
  }
  if (!replaceReady) {
    issues.push(issue.error(null, "No live GPU worker can run person replacement yet."));
  }
  if (referenceLimitMessage) {
    issues.push(issue.error(null, referenceLimitMessage));
  }
  if (audioOnlyReferenceSet) {
    issues.push(
      issue.error(
        null,
        "An audio reference can't be the only reference — add a reference image or video clip. Audio conditions the soundtrack, not the picture.",
      ),
    );
  }
  issues.push(...presetLoraIssues({ presetMissing, presetIncompatible, loraIncompatible, modelName }));
  return issues;
}
