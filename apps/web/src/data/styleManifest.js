// sc-13134 — Mechanical transform: the derived style catalog (parseStyleCatalog output,
// i.e. styles.json) → the backend manifest shape shipped at
// config/manifests/builtin.styles.jsonc. Kept as a single pure function so the generator
// (scripts/generate-styles.mjs) and the drift-guard test (data/styleCatalog.test.js) run
// the exact same derivation — the backend manifest can never hand-drift away from
// styles.json / documents/style.txt.
//
// The manifest is the SAME catalog the web app reads, reshaped only to the manifest
// envelope every builtin.*.jsonc uses: a `$schema` pointer + a `schemaVersion` (the
// catalog's `version`). The `source`, `promptTemplate`, and `groups` pass through
// verbatim so a headless/MCP client composing server-side sees byte-identical style text.

// Relative to the manifest's own location (config/manifests/), mirroring the
// recipe-preset manifest's `../../packages/schemas/recipe-preset.schema.json`.
export const STYLES_MANIFEST_SCHEMA = "../../packages/schemas/styles.schema.json";

/**
 * Reshape a parsed style catalog into the backend manifest object.
 * @param {{version:number, source:string, promptTemplate:string, groups:Array}} catalog
 * @returns {object} the builtin.styles.jsonc body
 */
export function catalogToStylesManifest(catalog) {
  return {
    $schema: STYLES_MANIFEST_SCHEMA,
    schemaVersion: catalog.version,
    source: catalog.source,
    promptTemplate: catalog.promptTemplate,
    groups: catalog.groups,
  };
}
