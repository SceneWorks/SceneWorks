import { readFileSync } from "node:fs";
import process from "node:process";
import JSON5 from "json5";
import old from "./old-catalog.json";

// APP-1's cutover replaces this adapter with its native ModelSpec projection, preserving the old
// fixture and its population. Passing a dump also checks live native descriptor/support changes.
export function currentCandidate() {
  if (process.env.CATALOG_CANDIDATE) return JSON.parse(readFileSync(process.env.CATALOG_CANDIDATE, "utf8"));
  const manifest = JSON5.parse(readFileSync("../../config/manifests/builtin.models.jsonc", "utf8"));
  // The capture dumper orders its controlled input population by id. Match that order in this
  // manifest bridge, using code-point comparison rather than host-dependent collation. Explicit
  // native dumps above retain their producer order, so a changed UI default remains observable.
  const models = manifest.models.map((model) => {
    const native = old.catalog.models.find((entry) => entry.id === model.id);
    return { ...model, ...(old.catalog.apiAddedCapabilities?.[model.id] ? { capabilities: [...new Set([...(model.capabilities ?? []), ...old.catalog.apiAddedCapabilities[model.id]])] } : {}), ...Object.fromEntries(["macSupport", "candleSupport", "preview", "decoders", "supportsLastOnlyImageToVideo"]
      .filter((key) => native?.[key] != null).map((key) => [key, native[key]])) };
  }).sort((left, right) => left.id < right.id ? -1 : left.id > right.id ? 1 : 0);
  return { ...old.catalog, models };
}
