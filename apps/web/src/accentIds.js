// Single source of truth for the accent-id list, derived from accents.js.
//
// The pre-paint theme script (theme-init.js) runs before the module graph loads
// and cannot import ES modules, so its copy of the accent ids is GENERATED from
// accents.js at build/dev time by the vite-plugin-theme-init plugin. This helper
// extracts the ids from the accents.js source text so both the plugin and the
// test that guards the two lists staying in sync share one parser.
//
// accents.js is the source of truth; editing it alone keeps the pre-paint list
// correct on the next dev-server start / build.

// Match `{ id: "teal", ... }` entries inside the exported ACCENTS array. Kept as
// a text parser (rather than importing accents.js) so it works in the vite
// config's Node context without transpiling the ES module.
const ACCENT_ID_RE = /\bid\s*:\s*["']([^"']+)["']/g;

/**
 * Extract the ordered list of accent ids from the source text of accents.js.
 * @param {string} accentsSource - raw contents of src/accents.js
 * @returns {string[]} accent ids in declaration order
 */
export function extractAccentIds(accentsSource) {
  const ids = [];
  let match;
  ACCENT_ID_RE.lastIndex = 0;
  while ((match = ACCENT_ID_RE.exec(accentsSource)) !== null) {
    ids.push(match[1]);
  }
  return ids;
}

// Marker in theme-init.template.js whose empty array the accent id list replaces.
// The template declares `const ACCENT_IDS = /* @accent-ids */ [];` — valid,
// lint-clean JS on its own — so the placeholder occurs exactly once.
const ACCENT_IDS_MARKER = /\/\* @accent-ids \*\/ \[\]/;

/**
 * Render the pre-paint theme-init.js from its template + the accent-id list.
 * Pure (no fs / URL) so it is testable outside the Vite/Node config context.
 * @param {string} templateSource - raw contents of theme-init.template.js
 * @param {string} accentsSource - raw contents of accents.js
 * @returns {string} the theme-init.js source with the id list substituted
 */
export function renderThemeInit(templateSource, accentsSource) {
  const ids = extractAccentIds(accentsSource);
  if (ids.length === 0) {
    throw new Error(
      "theme-init: no accent ids parsed from accents.js — the ACCENTS shape may have changed.",
    );
  }
  if (!ACCENT_IDS_MARKER.test(templateSource)) {
    throw new Error("theme-init: accent-ids marker not found in theme-init.template.js.");
  }
  return templateSource.replace(ACCENT_IDS_MARKER, JSON.stringify(ids));
}
