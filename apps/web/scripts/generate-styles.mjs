// Regenerate the derived style catalog from the authored documents/style.txt.
// Run via `npm run gen:styles` (from apps/web). Emits BOTH:
//   - apps/web/src/data/styles.json          (the web app's runtime catalog)
//   - config/manifests/builtin.styles.jsonc  (the backend/MCP catalog, sc-13134)
// from the SAME parse, so the web and backend catalogs can never drift. The
// styleCatalog.test.js drift guards fail CI if either committed artifact falls out
// of sync with style.txt, so this script is the only sanctioned way to update them.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { catalogToStylesManifest } from "../src/data/styleManifest.js";
import { parseStyleCatalog } from "../src/data/parseStyleCatalog.js";

const srcPath = fileURLToPath(new URL("../../../documents/style.txt", import.meta.url));
const dstPath = fileURLToPath(new URL("../src/data/styles.json", import.meta.url));
const manifestPath = fileURLToPath(
  new URL("../../../config/manifests/builtin.styles.jsonc", import.meta.url),
);

const raw = readFileSync(srcPath, "utf8");
const catalog = parseStyleCatalog(raw);
writeFileSync(dstPath, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");
writeFileSync(
  manifestPath,
  `${JSON.stringify(catalogToStylesManifest(catalog), null, 2)}\n`,
  "utf8",
);

const total = catalog.groups.reduce((n, g) => n + g.styles.length, 0);
console.log(
  `styles.json: ${catalog.groups.length} groups, ${total} styles → ${dstPath}`,
);
console.log(`builtin.styles.jsonc: mirror of styles.json → ${manifestPath}`);
