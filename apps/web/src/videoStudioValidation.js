// The Video Studio Generate gate in the app-wide vocabulary (epic 10644, sc-10650).
//
// This screen carried the epic's cleanest drift bug: `canSubmit` computed readiness from
// nine conditions, and a separate `blockedMessage` ternary re-derived a human-readable
// reason for five of them — two parallel expressions of the same rules, synced by hand.
// Here the reason and the gate are the same issue, so one cannot say "ready" while the
// other says why it isn't.

import { presetLoraIssues } from "./generationValidation.js";
import { issue } from "./validation/issues.js";

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
  modelName,
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
    issues.push(issue.error(null, "LTX video-conditioned generation needs an installed IC-LoRA preset."));
  }
  if (!replaceReady) {
    issues.push(issue.error(null, "No live GPU worker can run person replacement yet."));
  }
  issues.push(...presetLoraIssues({ presetMissing, presetIncompatible, loraIncompatible, modelName }));
  return issues;
}
