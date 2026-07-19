// sc-13130 — Runtime accessor for the shipped style catalog (styles.json). The catalog is a
// mechanical derivation of documents/style.txt (see parseStyleCatalog.js + styleCatalog.test.js
// for the drift guard); this module is the read-only lookup the Image Studio uses to render the
// grouped Style picker and to resolve a selected style id → its prompt text for composition.
//
// A style is identified everywhere in the studio by its `id`; the composer (styleComposer.js)
// wants the style's free-text `prompt`. `styleTextForId` is the one bridge between the two, so
// the selection state can stay a single string id (clean for saved-state + the sc-13132 recipe
// rehydration follow-on) while the payload fold still gets the prompt text it needs.
import catalog from "./styles.json";

// The 8 authored groups, each `{ id, name, description, styles: [{ id, name, prompt }] }`.
export const STYLE_GROUPS = catalog.groups;

// Flat id → style entry index, built once at module load. The style ids are unique across the
// whole catalog (parseStyleCatalog de-dupes with a global slug map), so one flat map is safe.
const STYLE_BY_ID = new Map();
for (const group of catalog.groups) {
  for (const style of group.styles) {
    STYLE_BY_ID.set(style.id, style);
  }
}

/**
 * Resolve a style id to its catalog entry (`{ id, name, prompt }`), or null when the id is
 * empty/unknown. Tolerant of null/undefined so callers can pass the raw saved-state value.
 */
export function findStyleById(id) {
  if (!id) {
    return null;
  }
  return STYLE_BY_ID.get(id) ?? null;
}

/**
 * The free-text `prompt` string for a style id, or null when no (valid) style is selected. This
 * is exactly the `styleText` the composer consumes; a null result means "pass-through" (no style).
 */
export function styleTextForId(id) {
  return findStyleById(id)?.prompt ?? null;
}
