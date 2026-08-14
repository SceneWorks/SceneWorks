import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

import { findActionPinViolations, formatViolations } from "./lib/action-pins.mjs";

const workflowDir = ".github/workflows";
const entries = (await readdir(workflowDir)).filter((name) => /\.ya?ml$/.test(name)).sort();

if (entries.length === 0) {
  console.error(`no workflows found in ${workflowDir}`);
  process.exit(1);
}

const violations = (
  await Promise.all(
    entries.map(async (name) => {
      const file = path.join(workflowDir, name);
      return findActionPinViolations(await readFile(file, "utf8"), file);
    }),
  )
).flat();

if (violations.length > 0) {
  console.error(
    `Every third-party action must be pinned to an immutable 40-character commit SHA with a trailing '# version' comment.\n${formatViolations(violations)}`,
  );
  process.exit(1);
}

console.log(`SceneWorks action pin check passed (${entries.length} workflows).`);
