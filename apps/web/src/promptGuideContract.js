// Single source of truth for WHICH catalog entries must ship a `ui.promptGuide`.
//
// Two independent authorities gate the manifest and used to disagree (sc-13783, epic 13678):
//   - scripts/check-scaffold.mjs (the web/scaffold CI gate) REQUIRED ui.promptGuide on EVERY
//     model entry.
//   - packages/schemas/model-manifest.schema.json treated ui.promptGuide as OPTIONAL for all.
// A schema-valid entry could therefore still RED the scaffold lane — a latent trap that sc-13684
// only papered over by hand-adding promptGuides to the new whisper/CLAP `type:"utility"` entries.
//
// The reconciled rule lives HERE and is consumed by BOTH authorities so neither can silently
// drift again:
//   - scripts/check-scaffold.mjs imports `promptGuideRequiredForModel` and skips exempt entries.
//   - the schema encodes the SAME exemption as an if/then conditional, and
//     promptGuideScaffoldSchemaContract.test.js asserts the schema's exemption set is byte-for-byte
//     the one declared here.
//
// A promptGuide is a PICKER-ONLY surface: only image/video/audio generation models reach the
// Image/Video Studio prompt entry that renders PromptGuideModal (selectedModel.ui.promptGuide).
// `type:"utility"` entries (whisper-base, clap-htsat-unfused, tile ControlNet, vision captioners,
// PiD checkpoints…) are provisioning/validation dependencies that never appear in a generation
// picker (generationModelsForType filters on an exact type match), so a user-facing prompt guide
// is meaningless for them and must NOT be required.

// Model `type`s that are provisioning/validation dependencies rather than user-facing generation
// pickers, and are therefore EXEMPT from the promptGuide requirement. Keep in lockstep with the
// schema's if/then exemption (the contract test enforces the match).
export const PROMPT_GUIDE_EXEMPT_TYPES = Object.freeze(["utility"]);

// True when a catalog entry must declare `ui.promptGuide`. Exempts the non-picker
// `type`s above; every other (picker) type is required to ship a guide.
export function promptGuideRequiredForModel(model) {
  return !PROMPT_GUIDE_EXEMPT_TYPES.includes(model?.type);
}
