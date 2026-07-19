import { describe, expect, it } from "vitest";

import { parseStyleCatalog, PROMPT_TEMPLATE } from "./parseStyleCatalog.js";
import { catalogToStylesManifest } from "./styleManifest.js";
import styles from "./styles.json";
// Raw source text (Vite `?raw`) so the test derives from the same bytes the
// generator reads. documents/style.txt lives outside the web root — see the
// server.fs.allow entry in vite.config.js (mirrors the license-corpus import).
import styleSource from "../../../../documents/style.txt?raw";
// sc-13134: the backend/MCP catalog. Read as ?raw + JSON.parse so the drift guard derives
// from the exact committed bytes. config/ is under the workspace root (searchForWorkspaceRoot
// in vite.config.js), so vite's server.fs.allow permits it like documents/.
import stylesManifestRaw from "../../../../config/manifests/builtin.styles.jsonc?raw";

// Guards sc-13127: styles.json must stay a mechanical derivation of
// documents/style.txt — never hand-edited. If style.txt changes, re-run
// `npm run gen:styles`; this test fails until styles.json is regenerated.
describe("style catalog: styles.json is derived from style.txt", () => {
  it("re-parsing style.txt reproduces the committed styles.json exactly", () => {
    expect(parseStyleCatalog(styleSource)).toEqual(styles);
  });
});

// Guards sc-13134: config/manifests/builtin.styles.jsonc — the catalog the backend embeds and
// serves at GET /api/v1/styles — must stay a mechanical derivation of style.txt, never hand-edited
// and never able to drift from the web's styles.json. The generator emits both from one parse; this
// asserts the committed manifest equals that same transform. If style.txt changes, re-run
// `npm run gen:styles`; this fails until builtin.styles.jsonc is regenerated.
describe("style catalog: builtin.styles.jsonc is derived from style.txt", () => {
  it("matches the manifest transform of the re-parsed style.txt", () => {
    const manifest = JSON.parse(stylesManifestRaw);
    expect(manifest).toEqual(catalogToStylesManifest(parseStyleCatalog(styleSource)));
  });

  it("mirrors the committed styles.json (same groups, version, template)", () => {
    const manifest = JSON.parse(stylesManifestRaw);
    expect(manifest).toEqual(catalogToStylesManifest(styles));
  });
});

describe("style catalog: structural invariants", () => {
  const allStyles = styles.groups.flatMap((g) => g.styles);

  it("has the eight top-level groups", () => {
    expect(styles.groups.map((g) => g.name)).toEqual([
      "Anime Style",
      "Cartoon Style",
      "Comics Style",
      "Drawing",
      "Photography",
      "Design",
      "Digital Painting",
      "Painting",
    ]);
  });

  it("ships the expected style count", () => {
    expect(allStyles.length).toBe(278);
  });

  it("carries the two-field prompt template", () => {
    expect(styles.promptTemplate).toBe(PROMPT_TEMPLATE);
    expect(styles.promptTemplate).toBe("Style: {style}\nDescription: {description}");
  });

  it("every group has a non-empty id, name, and description", () => {
    for (const g of styles.groups) {
      expect(g.id, `group ${g.name}`).toMatch(/\S/);
      expect(g.name, `group ${g.id}`).toMatch(/\S/);
      expect(g.description, `group ${g.name}`).toMatch(/\S/);
      expect(g.styles.length, `group ${g.name} styles`).toBeGreaterThan(0);
    }
  });

  it("every style has a non-empty id, name, and prompt", () => {
    for (const s of allStyles) {
      expect(s.id, `style ${s.name}`).toMatch(/\S/);
      expect(s.name, `style ${s.id}`).toMatch(/\S/);
      expect(s.prompt, `style ${s.name}`).toMatch(/\S/);
    }
  });

  it("style ids are globally unique across all groups", () => {
    const ids = allStyles.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("keeps the two same-name / different-text Film Noir entries as distinct ids", () => {
    const noir = allStyles.filter((s) => s.name.startsWith("Film Noir"));
    expect(noir.map((s) => s.id).sort()).toEqual(["film-noir", "film-noir-2"]);
    expect(noir[0].prompt).not.toBe(noir[1].prompt);
  });

  it("drops identical-text duplicates (Bande Dessinée appears once)", () => {
    const bande = allStyles.filter((s) => s.name.startsWith("Bande Dessinée"));
    expect(bande.length).toBe(1);
  });
});
