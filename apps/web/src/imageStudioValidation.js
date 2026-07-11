// The two CTA gates on Image Studio, in the app-wide validation vocabulary (epic 10644,
// sc-10649). Two independent buttons — single-image Generate and the prompt-batch Run —
// so two rule sets and two summaries. The batch panel's problems must never disable
// Generate, and vice versa.
//
// Both take pre-computed sub-results (presetValidation, validateCaption, missingKeys, …)
// rather than recomputing them: the screen already derives those for other uses, and a
// rule set is just the place they become blocking-or-not with a message. Pure functions
// of their inputs, so the requirement/error split is unit-testable.

import { presetLoraIssues } from "./generationValidation.js";
import { issue } from "./validation/issues.js";

// Single-image Generate. The conditions whose message already has a home stay OUT of
// here and remain plain gates in the button's `disabled` expression:
//   - a structured caption's field errors are listed inside StructuredPromptBuilder;
//   - a Mac capability block prints its own `.mac-gating-note`.
// What this surfaces is the preset/LoRA problems that used to be three separate
// `.inline-warning` paragraphs — now one chip row, one source with the button.
export function imageGenerateValidation({
  activeProject,
  structuredActive,
  captionHasContent,
  prompt,
  mode,
  characterId,
  presetMissing = [],
  presetIncompatible = [],
  loraIncompatible = [],
  modelName,
} = {}) {
  const issues = [];
  if (!activeProject) {
    issues.push(issue.requirement("project", "Open a project to generate"));
  }
  // A structured model needs caption content; everyone else a plain prompt. Both are
  // "you haven't written it yet" — silent, the empty field speaks for itself.
  if (structuredActive) {
    if (!captionHasContent) {
      issues.push(issue.requirement("caption", "Describe your shot in the builder"));
    }
  } else if (!prompt?.trim()) {
    issues.push(issue.requirement("prompt", "Write a prompt"));
  }
  if (mode === "character_image" && !characterId) {
    issues.push(issue.requirement("character", "Choose a character"));
  }
  issues.push(...presetLoraIssues({ presetMissing, presetIncompatible, loraIncompatible, modelName }));
  return issues;
}

// The prompt-batch Run. The errors are pushed in the same priority order the batch panel
// has always shown them one at a time, so `summarize().surfaced[0]` is exactly the message
// that used to win the `? :` chain. An empty batch is a silent requirement; its "add a
// prompt" hint is an empty-state affordance the panel renders on its own.
export function imageBatchValidation({
  activeProject,
  batchStructuredExpandBlocked,
  batchTotal,
  missingKeys = [],
  groupIssues = [],
  resolutionIssues = [],
  minDimension,
  maxDimension,
} = {}) {
  const issues = [];
  if (!activeProject) {
    issues.push(issue.requirement("project", "Open a project to run a batch"));
  }
  if (batchStructuredExpandBlocked) {
    issues.push(
      issue.error(
        null,
        "Batch on a structured-caption model needs the prompt-refiner model installed — it auto-writes a caption for each prompt.",
      ),
    );
  }
  if (missingKeys.length) {
    issues.push(issue.error(null, `Fill in a value for ${missingKeys.map((key) => `{{${key}}}`).join(", ")} to run.`));
  }
  if (groupIssues.length) {
    issues.push(
      issue.error(
        null,
        `Give each ${groupIssues.map((group) => `{{${group.label}:…}}`).join(", ")} the same number of options to run.`,
      ),
    );
  }
  if (resolutionIssues.length) {
    const res = resolutionIssues[0];
    issues.push(
      issue.error(null, `A prompt’s [${res.width}×${res.height}] size is out of range — each side must be ${minDimension}–${maxDimension}.`),
    );
  }
  if (batchTotal === 0) {
    issues.push(issue.requirement("prompts", "Add at least one prompt to run a batch."));
  }
  return issues;
}
