import { readdirSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// Desktop owns its own runner and dependencies. Generated output, dependencies, and
// agent checkouts are not repository tests; never follow symlinked directories.
const skipped = new Set([".git", ".claude", "node_modules", "target", "dist"]);
export function discoverRootTests(dir = root) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    const relative = path.relative(root, full);
    if (entry.isDirectory()) {
      return skipped.has(entry.name) || relative === path.join("apps", "desktop") ? [] : discoverRootTests(full);
    }
    return entry.isFile() && entry.name.endsWith(".test.mjs") ? [relative] : [];
  }).sort();
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const tests = discoverRootTests();
  if (!tests.length) throw new Error("No root Node tests discovered");
  const result = spawnSync(process.execPath, ["--test", ...tests], { cwd: root, stdio: "inherit" });
  if (result.error) throw result.error;
  process.exit(result.status ?? 1);
}
