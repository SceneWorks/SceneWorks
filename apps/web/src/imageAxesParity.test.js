import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import JSON5 from "json5";
import { describe, expect, it } from "vitest";
import { fallbackModels } from "./constants.js";

// Manifest ↔ constants parity for the `image` capability sub-block (sc-15299).
//
// Image Studio hides Guidance / Negative prompt for an engine that declares
// `image.supportsGuidance` / `image.supportsNegativePrompt` false. At runtime that comes from the
// live catalog (seeded from config/manifests/builtin.models.jsonc), but `constants.js`
// (`fallbackModels`) carries a HAND-MIRRORED copy used BEFORE the catalog loads. Without this
// guard the mirror silently drifts and a CFG-free model briefly renders both dead controls on a
// cold start. Same shape as controlModesParity.test.js.
//
// ⚠️ Polarity: an ABSENT `image` block means BOTH axes are supported — the opposite of `audio`.
// So the reverse direction matters just as much: a mirror entry must not declare an axis absent
// that the manifest says is present, or the seed would hide a live control.
const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");

function loadManifestModels() {
  const parsed = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8"));
  const models = Array.isArray(parsed) ? parsed : parsed.models;
  expect(Array.isArray(models), "manifest must expose a models array").toBe(true);
  return models;
}

describe("image axes ↔ manifest parity (sc-15299)", () => {
  const manifestModels = loadManifestModels();
  const manifestById = new Map(manifestModels.map((model) => [model.id, model]));
  const seededImageModels = fallbackModels.filter((model) => model.type === "image");

  it("mirrors the manifest `image` block for every seeded image model, in both directions", () => {
    expect(seededImageModels.length, "the seed carries image models to check").toBeGreaterThan(0);
    for (const fallback of seededImageModels) {
      const manifest = manifestById.get(fallback.id);
      if (!manifest) {
        // A seed-only entry (no manifest twin) has nothing to mirror.
        continue;
      }
      expect(
        fallback.image ?? null,
        `${fallback.id}: the pre-catalog seed must declare the same generation axes as the manifest`,
      ).toEqual(manifest.image ?? null);
    }
  });

  it("keeps at least one CFG-free and one negative-free entry in the mirror", () => {
    // Guards the guard: if the seed ever lost every declaring entry the test above would pass
    // vacuously. These two shapes are what the polarity actually rides on.
    const cfgFree = seededImageModels.filter((model) => model.image?.supportsGuidance === false);
    const negativeOnly = seededImageModels.filter(
      (model) => model.image?.supportsNegativePrompt === false && model.image?.supportsGuidance === undefined,
    );
    expect(cfgFree.map((model) => model.id)).toContain("flux_schnell");
    expect(negativeOnly.map((model) => model.id)).toContain("flux_dev");
  });

  it("declares no `image` block on a seeded model that takes both axes", () => {
    // Absent means TRUE: every guidance-taking seed entry must stay silent.
    for (const fallback of seededImageModels) {
      const manifest = manifestById.get(fallback.id);
      if (manifest && manifest.image == null) {
        expect(fallback.image, `${fallback.id} takes both axes and must declare nothing`).toBeUndefined();
      }
    }
  });
});
