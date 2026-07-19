// Regenerate apps/web/src/data/styles.json from the authored documents/style.txt.
// Run via `npm run gen:styles` (from apps/web). The styleCatalog.test.js drift
// guard fails CI if the committed styles.json ever falls out of sync with
// style.txt, so this script is the only sanctioned way to update it.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { parseStyleCatalog } from "../src/data/parseStyleCatalog.js";

const srcPath = fileURLToPath(new URL("../../../documents/style.txt", import.meta.url));
const dstPath = fileURLToPath(new URL("../src/data/styles.json", import.meta.url));

const raw = readFileSync(srcPath, "utf8");
const catalog = parseStyleCatalog(raw);
writeFileSync(dstPath, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");

const total = catalog.groups.reduce((n, g) => n + g.styles.length, 0);
console.log(
  `styles.json: ${catalog.groups.length} groups, ${total} styles → ${dstPath}`,
);
