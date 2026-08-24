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
// stops that mirror from silently drifting from the manifest. This test is the guard: every accepted
// SDXL OpenPose backbone must be present in both surfaces with the complete matching pose UI; other
// fallback control entries must still mirror their manifest modes and scale exactly. Missing entries or
// field drift fail here instead of shipping a picker that offers the wrong controls.
const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");
const EPIC_20738_SDXL_OPENPOSE_IDS = [
  "illustrious_xl_v1",
  "illustrious_xl_v2",
  "realvisxl",
  "realvisxl_lightning",
  "sdxl",
];
const EPIC_20738_OPENPOSE_SCALE = {
  label: "Control strength",
  default: 1.0,
  min: 0.0,
  max: 2.0,
  step: 0.05,
};

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

  it("the manifest advertises strict control on the expected backbones", () => {
    const ids = manifestControlModels.map((model) => model.id).sort();
    // Fun-Union/Krea plus the exact five SDXL OpenPose backbones accepted on both native providers.
    expect(ids).toEqual([
      "flux2_dev",
      "flux_dev",
      "illustrious_xl_v1",
      "illustrious_xl_v2",
      "krea_2_turbo",
      "qwen_image",
      "realvisxl",
      "realvisxl_lightning",
      "sdxl",
      "z_image",
      "z_image_turbo",
    ]);
  });

  it("exposes the complete OpenPose picker only for epic 20738's exact five SDXL backbones", () => {
    const exposed = manifestModels
      .filter(
        (model) =>
          model.family === "sdxl" &&
          model.ui?.poseLibrary === true &&
          model.ui?.poseControlScale === true &&
          Array.isArray(model.ui?.controlModes) &&
          model.ui.controlModes.includes("pose"),
      )
      .sort((left, right) => left.id.localeCompare(right.id));

    expect(exposed.map((model) => model.id)).toEqual(EPIC_20738_SDXL_OPENPOSE_IDS);
    for (const model of exposed) {
      expect(model.ui.controlModes, `${model.id} must stay pose-only`).toEqual(["pose"]);
      expect(model.ui.controlScale, `${model.id} must expose the accepted OpenPose scale`).toEqual(
        EPIC_20738_OPENPOSE_SCALE,
      );
    }
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

  it("mirrors the complete pose UI for every accepted SDXL OpenPose backbone", () => {
    for (const id of EPIC_20738_SDXL_OPENPOSE_IDS) {
      const manifest = manifestById.get(id);
      const fallback = fallbackModels.find((model) => model.id === id);
      expect(manifest, `${id} must exist in the manifest`).toBeTruthy();
      expect(fallback, `${id} must exist in fallbackModels`).toBeTruthy();
      for (const field of ["poseLibrary", "poseControlScale", "controlModes", "controlScale"]) {
        expect(fallback.ui?.[field], `${id} ${field} must mirror the manifest`).toEqual(
          manifest.ui[field],
        );
      }
    }
  });

  it("seeds RealVisXL Lightning with its exact identity and five-step CFG policy", () => {
    const manifest = manifestById.get("realvisxl_lightning");
    const fallback = fallbackModels.find((model) => model.id === "realvisxl_lightning");
    for (const field of ["name", "family", "type", "capabilities", "defaults"]) {
      expect(fallback?.[field], `realvisxl_lightning ${field} must mirror the manifest`).toEqual(
        manifest?.[field],
      );
    }
    expect(fallback.ui.description).toMatch(/~5 steps/);
    expect(fallback.ui.description).toMatch(/CFG-free.*guidance 1\.0/i);
    expect(fallback.ui.description).toMatch(/negative prompt/i);
  });
});
