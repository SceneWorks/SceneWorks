import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import JSON5 from "json5";
import previewSupport from "../data/previewSupport.json";
import { projectBuiltinMemoryEvidence } from "../../../../scripts/lib/builtin-memory-evidence.mjs";

// Test-only projection of the manifest shipped by the API. It is never used to
// infer installation or availability, which are supplied by the live catalog.
// Measured memory facts live in registry/evidence.json (SW-7031); the API
// projects them back onto the product manifest, and so does this loader.
const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
const manifest = projectBuiltinMemoryEvidence(
  JSON5.parse(readFileSync(resolve(root, "config/manifests/builtin.models.jsonc"), "utf8")),
  JSON.parse(readFileSync(resolve(root, "config/manifests/registry/evidence.json"), "utf8")),
);
if (!Array.isArray(manifest.models)) throw new Error("Builtin manifest must contain models");
export const fallbackModels = manifest.models.map((model) => {
  const byBackend = previewSupport.models[model.id];
  return byBackend ? { ...model, preview: { byBackend: { ...byBackend } } } : model;
});
