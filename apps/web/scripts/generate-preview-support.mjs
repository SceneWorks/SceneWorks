// Regenerate the derived live-preview support catalog (`SW-6983`, `REG-3`, `M8`).
// Run via `npm run gen:preview-support` (from apps/web). Emits BOTH:
//   - config/manifests/builtin.preview-support.jsonc  (the backend catalog block sceneworks-core
//                                                      embeds and rust-api merges onto /api/v1/models)
//   - apps/web/src/data/previewSupport.json           (test-only preview support data)
// purely from:
//   1. config/engine-capabilities/sga/{macos-mlx,candle-cuda}.json (`sga.capabilities/2`)
//   2. crates/sceneworks-worker/src/engines.rs (`MODEL_TABLE` product->engine join)
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  assertBackendCoverage,
  catalogToWebPreviewSupport,
  derivePreviewSupport,
  parseBespokeRouteAliases,
  parseEngineModelTable,
  REQUIRED_AUDIO_BACKENDS,
  REQUIRED_MEDIA_BACKENDS,
  REQUIRED_SGA_EXPORTS,
} from "../src/data/previewSupportDerivation.js";

const sgaDir = fileURLToPath(
  new URL("../../../config/engine-capabilities/sga", import.meta.url),
);
const enginesPath = fileURLToPath(
  new URL("../../../crates/sceneworks-worker/src/engines.rs", import.meta.url),
);
const routeBindingsPath = fileURLToPath(
  new URL("../../../crates/sceneworks-core/src/registry/route_bindings.rs", import.meta.url),
);
const manifestPath = fileURLToPath(
  new URL("../../../config/manifests/builtin.preview-support.jsonc", import.meta.url),
);
const webPath = fileURLToPath(new URL("../src/data/previewSupport.json", import.meta.url));

/** Read the required desktop/server `sga.capabilities/2` exports. */
export function readSgaExports(dir = sgaDir) {
  return REQUIRED_SGA_EXPORTS.map(({ file }) => JSON.parse(readFileSync(`${dir}/${file}`, "utf8")));
}

const sgaExports = readSgaExports();
assertBackendCoverage(
  REQUIRED_MEDIA_BACKENDS,
  sgaExports.map((doc) => doc.backend),
  "media",
);
assertBackendCoverage(
  REQUIRED_AUDIO_BACKENDS,
  [...new Set(sgaExports.map((doc) => doc.audio_backend).filter(Boolean))],
  "audio",
);

const catalog = derivePreviewSupport(
  [
    ...parseEngineModelTable(readFileSync(enginesPath, "utf8")),
    ...parseBespokeRouteAliases(readFileSync(routeBindingsPath, "utf8")),
  ],
  sgaExports,
);

writeFileSync(manifestPath, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");
writeFileSync(
  webPath,
  `${JSON.stringify(catalogToWebPreviewSupport(catalog), null, 2)}\n`,
  "utf8",
);

const ids = Object.keys(catalog.models);
const supporting = ids.filter((id) =>
  Object.values(catalog.models[id]).some((supported) => supported === true),
);
console.log(
  `builtin.preview-support.jsonc: ${ids.length} models over backends [${catalog.backends.join(", ")}]` +
    ` -> ${manifestPath}`,
);
console.log(`previewSupport.json: mirror for the web fallback -> ${webPath}`);
console.log(`live preview advertised by: ${supporting.join(", ") || "(none)"}`);
