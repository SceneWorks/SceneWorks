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
// sc-13366: the per-style tailored subject prompts (also used to generate the preview thumbnails,
// sc-13135). Surfaced in the Image Studio as a single per-style "Try:" hint when a style is selected.
import thumbnailPrompts from "./styleThumbnailPrompts.json";

// The 8 authored groups, each `{ id, name, description, styles: [{ id, name, prompt }] }`.
export const STYLE_GROUPS = catalog.groups;

// Flat id → style entry index, built once at module load. The style ids are unique across the
// whole catalog (parseStyleCatalog de-dupes with a global slug map), so one flat map is safe.
const STYLE_BY_ID = new Map();
// Group id → group entry. sc-13171 makes the group-level generic style selectable: the two-level
// picker stores the GROUP id (e.g. "anime-style") when the user picks a group's "overall" style,
// and the composer resolves that id to the group's `description` text — distinct from any sub-style.
const GROUP_BY_ID = new Map();
for (const group of catalog.groups) {
  GROUP_BY_ID.set(group.id, group);
  for (const style of group.styles) {
    STYLE_BY_ID.set(style.id, style);
  }
}

// Invariant guard (sc-13171): group ids and sub-style ids share ONE id-space because a single
// stored `styleId` must resolve unambiguously to exactly one of them. If the derived catalog ever
// introduced a group id that also names a sub-style, `styleTextForId` could not tell which text to
// return, so we fail loudly at module load rather than silently pick one. The styleCatalog tests
// assert this holds for the shipped data; this is the runtime backstop.
for (const groupId of GROUP_BY_ID.keys()) {
  if (STYLE_BY_ID.has(groupId)) {
    throw new Error(
      `styleCatalog: group id "${groupId}" collides with a sub-style id — style ids must be globally unique so a stored styleId is unambiguous.`,
    );
  }
}

/**
 * Resolve a style id to a catalog entry, or null when the id is empty/unknown. Tolerant of
 * null/undefined so callers can pass the raw saved-state value.
 *
 * A sub-style id resolves to its exact `{ id, name, prompt }` entry. A GROUP id (sc-13171)
 * resolves to a synthetic "general" entry `{ id, name, prompt, isGroup: true, groupId }` whose
 * `prompt` is the group's top-level `description` — enough for breadcrumb/label rendering and for
 * `styleTextForId` to fold the broad group style into the payload.
 */
export function findStyleById(id) {
  if (!id) {
    return null;
  }
  const style = STYLE_BY_ID.get(id);
  if (style) {
    return style;
  }
  const group = GROUP_BY_ID.get(id);
  if (group) {
    return { id: group.id, name: group.name, prompt: group.description, isGroup: true, groupId: group.id };
  }
  return null;
}

/**
 * The free-text `prompt` string for a style id, or null when no (valid) style is selected. This
 * is exactly the `styleText` the composer consumes; a null result means "pass-through" (no style).
 * Resolves BOTH a sub-style id (→ its `prompt`) and a group id (→ that group's `description`), so
 * the preview (sc-13131) and the payload fold stay byte-for-byte in sync through the same bridge.
 */
export function styleTextForId(id) {
  return findStyleById(id)?.prompt ?? null;
}

/**
 * The tailored subject prompt for a style id (the same prompt used to generate that style's preview
 * thumbnail — styleThumbnailPrompts.json, sc-13135), or null when the id is empty/unmapped. The
 * Image Studio surfaces this as a single per-style "Try:" hint (sc-13366): a strong, style-fitting
 * starting prompt. Resolves both sub-style ids and group ids (the map covers all 286 catalog ids).
 */
export function styleHintForId(id) {
  if (!id) {
    return null;
  }
  const hint = thumbnailPrompts.prompts?.[id];
  return typeof hint === "string" && hint.trim() ? hint : null;
}
