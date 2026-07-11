// Booru quality-prefix + negative seeding for the Image Studio prompt box (sc-10760). Anima and
// Illustrious are danbooru-tag models: they render low-effort art from a bare natural-language sentence
// and reward a quality prefix (`masterpiece, best quality, …`) plus tag-style prompting — exactly what
// their model cards prescribe. The studio seeds the model-declared `ui.defaultPrompt` into an UNEDITED
// prompt box and `ui.defaultNegativePrompt` into an EMPTY negative box, and shows `ui.promptHint` under
// the box. These pure helpers hold the decisions so they can be unit-tested without rendering the studio.

// The generic scene default the prompt box starts on. A model WITHOUT a `defaultPrompt` restores this in
// an unedited box, so a booru quality prefix never lingers after the user switches to a non-booru model.
export const DEFAULT_SCENE_PROMPT = "A cinematic frame of a neon street at midnight";

// The prompt to seed into an UNEDITED text-to-image prompt box for a model's `ui`. Returns the model's
// booru quality prefix when it declares a non-empty string `defaultPrompt`; otherwise the scene default.
export function promptSeedFor(modelUi) {
  const prefix = modelUi?.defaultPrompt;
  return typeof prefix === "string" && prefix ? prefix : DEFAULT_SCENE_PROMPT;
}

// Whether the studio seeds a curated default negative into an EMPTY box in `mode`. Character mode has
// always done this (sc-3857); text-to-image now does too (sc-10760) so booru models get their booru
// negative there. Edit mode keeps whatever the user has.
export function seedsNegativeInMode(mode) {
  return mode === "character_image" || mode === "text_to_image";
}

// The booru prompt hint to render under the box for a model's `ui`, or null when none is declared.
export function promptHintFor(modelUi) {
  const hint = modelUi?.promptHint;
  return typeof hint === "string" && hint ? hint : null;
}
