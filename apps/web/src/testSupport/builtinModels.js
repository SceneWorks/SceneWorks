import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import JSON5 from "json5";
import previewSupport from "../data/previewSupport.json";

// Test-only projection of the manifest shipped by the API. It is never used to
// infer installation or availability, which are supplied by the live catalog.
const manifest = JSON5.parse(readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "../../../../config/manifests/builtin.models.jsonc"), "utf8"));
if (!Array.isArray(manifest.models)) throw new Error("Builtin manifest must contain models");
export const fallbackModels = manifest.models.map((model) => {
  const byBackend = previewSupport.models[model.id];
  return byBackend ? { ...model, preview: { byBackend: { ...byBackend } } } : model;
});
