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
import { MAX_IMAGE_DIMENSION, MIN_IMAGE_DIMENSION } from "./resolutionOverride.js";
import { promptBudget } from "./styleComposer.js";
import { issue } from "./validation/issues.js";

export function batchPromptBudgetOverages(composedPrompts = []) {
  const overages = [];
  let index = 0;
  for (const prompt of composedPrompts) {
    const budget = promptBudget(prompt);
    if (budget.over) overages.push({ item: index + 1, ...budget });
    index += 1;
  }
  return overages;
}

export function batchPromptBudgetMessage(overages = []) {
  const items = overages.map(({ item, length, max }) => `${item} (${length}/${max})`).join(", ");
  return `Batch prompt${overages.length === 1 ? "" : "s"} ${items} exceed${
    overages.length === 1 ? "s" : ""
  } the character limit — shorten the prompt or pick a shorter style.`;
}

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
  editSourceMissing = false,
  editLoraMissing = false,
  // sc-13133 / sc-13224: the COMPOSED outgoing prompt and whether a Style Catalog entry is active.
  // `composedPrompt` is the exact string that will be sent — the Subject:/Style: composition for a
  // prose model, or the style-injected re-serialized caption for a structured model — so the cap is
  // measured on IT, not the raw prompt field: a ~700–900 char style wrapped around a long-but-under-cap
  // prompt, or merged into a caption, can push past 4000.
  styleActive = false,
  composedPrompt = "",
  presetMissing = [],
  presetIncompatible = [],
  loraIncompatible = [],
  modelName,
  // Krea 2 multi-phase denoise (epic 13879 S5, sc-13885): pre-computed issues from
  // imageMultiPhase.multiPhaseIssues (an enabled-but-broken phase list blocks Generate with an
  // error). Empty when the editor is off or the model has no multi-phase lane.
  multiPhaseIssues = [],
  // Free Width/Height override out of the selected model's native range or off its required stride
  // (sc-14058, native-resolution families like Mage-Flow). A pre-computed, user-facing ERROR string
  // (or null) — a value the user actively broke that nothing else on the form explains, so it blocks
  // Generate AND is surfaced. Null/absent leaves the gate unchanged for every other model.
  dimensionError = null,
  // A restored alternate decoder cannot be reconciled or safely omitted until the platform
  // capability response identifies the active engine. Native selections never set this gate.
  decoderCapabilitiesPending = false,
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
  // Composed-prompt budget guard (sc-13133 / sc-13224). ONLY when a style is active: styleless
  // behavior is unchanged (the raw prompt keeps whatever gating it had — the backend still bounds it,
  // but the studio doesn't pre-flag it). Applies to BOTH prose and structured models now that the
  // Style axis merges into a structured caption too (sc-13224): `composedPrompt` is the exact string
  // sent in either case, and injecting a style can push a caption over the cap. An error, not a silent
  // requirement: nothing else on the form explains why Generate is dead. The message names the actual
  // numbers and the two ways out.
  if (styleActive && composedPrompt) {
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
  if (mode === "character_image" && !characterId) {
    issues.push(issue.requirement("character", "Choose a character"));
  }
  // Edit mode needs a source image to edit — silent, the empty source picker is the affordance.
  if (editSourceMissing) {
    issues.push(issue.requirement("source", "Select a source image to edit"));
  }
  // A Krea-style edit LoRA that isn't downloaded yet (epic 10871): a silent gate — the edit
  // source band renders the actionable "required to edit — Download" note, so this only blocks
  // Generate without duplicating that message.
  if (editLoraMissing) {
    issues.push(issue.requirement("editLora", "Download the required edit LoRA to run edits"));
  }
  issues.push(...presetLoraIssues({ presetMissing, presetIncompatible, loraIncompatible, modelName }));
  // Multi-phase denoise problems (sc-13885) — already kinded (errors) by multiPhaseIssues.
  issues.push(...multiPhaseIssues);
  // A broken free Width/Height override (sc-14058): an error, not a requirement — the two number boxes
  // hold a value the user actively broke, and only this message explains the dead Generate button.
  if (dimensionError) {
    issues.push(issue.error(null, dimensionError));
  }
  if (decoderCapabilitiesPending) {
    issues.push(
      issue.error(
        "decoder",
        "Waiting for engine capabilities before using the restored decoder.",
      ),
    );
  }
  return issues;
}

// The user-facing ERROR for an inline [WxH] batch directive that breaks the selected model's geometry
// (sc-14058). Names the offending size, then the model's own range — and its required stride when the
// model renders at native resolution (a declared step > 1, e.g. Mage-Flow's ÷16). A non-native model
// (step <= 1) keeps the exact "out of range — each side must be MIN–MAX" wording it always had, so those
// models don't regress. This is the batch-path twin of dimensionConstraintMessage for the single path.
export function batchResolutionMessage({ width, height } = {}, { min, max, step } = {}) {
  const size = `[${width}×${height}]`;
  if (Number.isInteger(step) && step > 1) {
    return `A prompt’s ${size} size isn’t valid for this model — each side must be ${min}–${max} and a multiple of ${step}.`;
  }
  return `A prompt’s ${size} size is out of range — each side must be ${min}–${max}.`;
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
  promptBudgetOverages = [],
  // Per-model geometry constraints for the inline [WxH] batch directive (sc-14058). Same source the
  // single-generate gate reads — modelDimensionConstraints(selectedModel) — so a native-resolution model
  // (Mage-Flow) narrows the batch check to its own range + ÷16 stride, while every other model keeps the
  // global 256–4096 / no-stride envelope. Defaults to that global envelope for callers with no model.
  dimensionConstraints = { min: MIN_IMAGE_DIMENSION, max: MAX_IMAGE_DIMENSION, step: 1 },
  decoderCapabilitiesPending = false,
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
    issues.push(issue.error(null, batchResolutionMessage(resolutionIssues[0], dimensionConstraints)));
  }
  if (promptBudgetOverages.length) {
    issues.push(issue.error(null, batchPromptBudgetMessage(promptBudgetOverages)));
  }
  if (decoderCapabilitiesPending) {
    issues.push(
      issue.error(
        "decoder",
        "Waiting for engine capabilities before using the restored decoder.",
      ),
    );
  }
  if (batchTotal === 0) {
    issues.push(issue.requirement("prompts", "Add at least one prompt to run a batch."));
  }
  return issues;
}
