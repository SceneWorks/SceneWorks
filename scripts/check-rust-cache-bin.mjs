import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

import {
  findRustCacheBinViolations,
  formatRustCacheBinViolations,
} from "./lib/rust-cache-bin.mjs";

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
      return findRustCacheBinViolations(await readFile(file, "utf8"), file);
    }),
  )
).flat();

if (violations.length > 0) {
  console.error(
    `Swatinem/rust-cache on a SELF-HOSTED runner must set 'cache-bin: "false"'.\n` +
      `Its post step deletes every regular file in $CARGO_HOME/bin that is absent from\n` +
      `.crates2.json — including rustup itself — which is harmless on a throwaway VM and\n` +
      `destructive where CARGO_HOME is a real user profile: the whole Rust toolchain\n` +
      `disappears for every process on the box, with no red build anywhere.\n` +
      `See scripts/lib/rust-cache-bin.mjs and inference PR #467.\n` +
      `${formatRustCacheBinViolations(violations)}`,
  );
  process.exit(1);
}

console.log(`SceneWorks rust-cache cache-bin check passed (${entries.length} workflows).`);
