// Validation shared by both studios' Generate gates (epic 10644). The preset/LoRA
// compatibility problems are identical in Image and Video Studio — same three messages
// the `PresetValidationWarnings` paragraphs and the selected-LoRA `.inline-warning`
// used to render — so they live here rather than drifting apart in two files.

import { issue } from "./validation/issues.js";

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
