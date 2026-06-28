import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import JSON5 from "json5";
import { describe, expect, it } from "vitest";
import { fallbackModels } from "./constants.js";

// Manifest ↔ constants parity for strict-control modes (sc-8245, folded in from the sc-8244 review).
//
// The web control panel gates the pose/canny/depth picker off the selected model's `ui.controlModes`
// (and binds the scale slider to `ui.controlScale`). At runtime those come from the live catalog —
// itself seeded from `config/manifests/builtin.models.jsonc` — but `apps/web/src/constants.js`
// (`fallbackModels`) carries a HAND-MIRRORED copy used before the catalog loads (and in tests). Nothing
// stops that mirror from silently drifting from the manifest. This test is the guard: for every backbone
// the manifest advertises strict control on AND that also appears in `fallbackModels`, the mirror's
// `controlModes` + `controlScale` must match the manifest exactly. A manifest change that isn't mirrored
// (or a mirror typo) fails here instead of shipping a picker that offers the wrong modes / scale defaults.
const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");

function loadManifestModels() {
  const raw = readFileSync(MANIFEST_PATH, "utf8");
  const parsed = JSON5.parse(raw);
  const models = Array.isArray(parsed) ? parsed : parsed.models;
  expect(Array.isArray(models), "manifest must expose a models array").toBe(true);
  return models;
}

describe("controlModes ↔ manifest parity (sc-8245)", () => {
  const manifestModels = loadManifestModels();
  const manifestById = new Map(manifestModels.map((model) => [model.id, model]));

  // The manifest backbones that advertise strict control — the authority the picker gates on.
  const manifestControlModels = manifestModels.filter(
    (model) => Array.isArray(model?.ui?.controlModes) && model.ui.controlModes.length > 0,
  );

  it("the manifest advertises strict control on the five Fun-Union backbones", () => {
    const ids = manifestControlModels.map((model) => model.id).sort();
    expect(ids).toEqual(["flux2_dev", "flux_dev", "qwen_image", "z_image", "z_image_turbo"]);
  });

  it.each(
    fallbackModels
      .filter((model) => Array.isArray(model?.ui?.controlModes) && model.ui.controlModes.length > 0)
      .map((model) => [model.id, model]),
  )("constants.js %s mirrors the manifest controlModes + controlScale", (id, fallback) => {
    const manifest = manifestById.get(id);
    expect(manifest, `${id} must exist in the manifest`).toBeTruthy();
    // controlModes must match exactly, including order (the picker renders in this order).
    expect(fallback.ui.controlModes).toEqual(manifest.ui.controlModes);
    // controlScale (label/default/min/max/step) is the slider config — it must match too.
    expect(fallback.ui.controlScale).toEqual(manifest.ui.controlScale);
  });

  it("every manifest control backbone present in fallbackModels mirrors controlModes", () => {
    // The reverse guard: a manifest control backbone that fallbackModels DOES carry (by id) but with a
    // missing/empty controlModes would silently drop the picker — catch that drift too. Backbones absent
    // from the seed list (e.g. flux2_dev, which loads from the live catalog) are intentionally skipped.
    for (const manifest of manifestControlModels) {
      const fallback = fallbackModels.find((model) => model.id === manifest.id);
      if (!fallback) {
        continue;
      }
      expect(fallback.ui?.controlModes, `${manifest.id} controlModes must be mirrored`).toEqual(
        manifest.ui.controlModes,
      );
      expect(fallback.ui?.controlScale, `${manifest.id} controlScale must be mirrored`).toEqual(
        manifest.ui.controlScale,
      );
    }
  });
});
