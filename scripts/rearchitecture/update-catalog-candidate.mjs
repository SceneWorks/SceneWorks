#!/usr/bin/env node
// Refresh the current native candidate. The pinned old catalog is never read or rewritten here.
import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { renderBaseline } from "./baseline-json.mjs";

const root = fileURLToPath(new URL("../../", import.meta.url));
const output = fileURLToPath(new URL("../../apps/web/src/testUtils/catalogBaseline/candidate-catalog.json", import.meta.url));
const dump = execFileSync("cargo", ["run", "--locked", "-q", "-j2", "-p", "sceneworks-core", "--bin", "dump-ui-catalog"], {
  cwd: root,
  encoding: "utf8",
  maxBuffer: 16 * 1024 * 1024,
});
writeFileSync(output, `${renderBaseline(JSON.parse(dump))}\n`);
console.log(`Updated native candidate at ${output}`);
