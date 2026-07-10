// Validation shared by both studios' Generate gates (epic 10644). The preset/LoRA
// compatibility problems are identical in Image and Video Studio — same three messages
// the `PresetValidationWarnings` paragraphs and the selected-LoRA `.inline-warning`
// used to render — so they live here rather than drifting apart in two files.

import { issue } from "./validation/issues.js";

// The studios' inline "Save as Preset" dialog (epic 10644, sc-10651). A blank name is a
// silent requirement; a mode the current studio can't save is an error whose message the
// caller already supplies as a tooltip (`saveTitle`) — routed here so the reason is
// always visible, not only on hover.
export function savePresetDialogValidation({ presetName, saveDisabled, saveTitle } = {}) {
  const issues = [];
  if (!presetName?.trim()) {
    issues.push(issue.requirement("name", "Name this setup"));
  }
  if (saveDisabled) {
    issues.push(issue.error(null, saveTitle ?? "This mode can’t be saved as a preset."));
  }
  return issues;
}

export function presetLoraIssues({ presetMissing = [], presetIncompatible = [], loraIncompatible = [], modelName } = {}) {
  const issues = [];
  const model = modelName ?? "the selected model";
  if (presetMissing.length) {
    issues.push(
      issue.error(
        null,
        `Preset cannot run until LoRA import finishes: ${presetMissing.join(", ")}. Wait for the Queue or choose another preset.`,
      ),
    );
  }
  if (presetIncompatible.length) {
    issues.push(
      issue.error(
        null,
        `Preset cannot run with ${model} because these LoRAs are incompatible: ${presetIncompatible.join(", ")}. Choose another preset or model.`,
      ),
    );
  }
  if (loraIncompatible.length) {
    issues.push(
      issue.error(
        null,
        `Generate is blocked because these selected LoRAs are incompatible with ${model}: ${loraIncompatible.join(", ")}.`,
      ),
    );
  }
  return issues;
}
