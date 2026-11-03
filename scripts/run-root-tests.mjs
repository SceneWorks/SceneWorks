import { readdirSync } from "node:fs";
import path from "node:path";
import { availableParallelism } from "node:os";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// Desktop owns its own runner and dependencies. Generated output, dependencies, and
// agent checkouts are not repository tests; never follow symlinked directories.
const skipped = new Set([".git", ".claude", "node_modules", "target", "dist"]);
export function discoverRootTests(dir = root) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    const relative = path.relative(root, full).split(path.sep).join("/");
    if (entry.isDirectory()) {
      return skipped.has(entry.name) || relative === "apps/desktop" ? [] : discoverRootTests(full);
    }
    return entry.isFile() && entry.name.endsWith(".test.mjs") ? [relative] : [];
  }).sort();
}
export function rootTestConcurrency(parallelism = availableParallelism()) {
  return Math.min(4, parallelism);
}
// Keep this suite in its original invocation after all parallel tests finish.
export const SERIAL_TEST = "scripts/matrix-removed-sources.test.mjs";
export function rootTestGroups(files = discoverRootTests()) {
  if (files.filter((file) => file === SERIAL_TEST).length !== 1 || new Set(files).size !== files.length) {
    throw new Error("Root test inventory is duplicated or missing its serial suite");
  }
  const parallel = files.filter((file) => file !== SERIAL_TEST);
  if (!parallel.length) throw new Error("No parallel root Node tests discovered");
  return [parallel, [SERIAL_TEST]];
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  for (const tests of rootTestGroups()) {
    const result = spawnSync(process.execPath, ["--test", `--test-concurrency=${rootTestConcurrency()}`, ...tests], { cwd: root, stdio: "inherit" });
    if (result.error) throw result.error;
    if (result.status !== 0) { process.exitCode = result.status ?? 1; break; }
  }
}
