// The full npm check command remains the single test inventory. Split only its existing files;
// no per-test weakening, no optional source checks, and no second hand-maintained portable list.
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

export const HISTORICAL_TESTS = [
  "scripts/anchor-loader-closure.test.mjs",
  "scripts/measure-memory-catalog.test.mjs",
  "scripts/manifest-memory-declarations.test.mjs",
  "scripts/starvector-terminal-producer.test.mjs",
];

export function scaffoldTestGroups(command, mode) {
  if (!["portable", "history"].includes(mode)) throw new Error("expected --portable or --history");
  const commands = command.split("&&").map((part) => part.trim().split(/\s+/));
  if (commands.some((parts) => parts[0] !== "node" || parts[1] !== "--test" || parts.slice(2).some((file) => !/^scripts\/[\w/.-]+\.test\.mjs$/.test(file)))) {
    throw new Error("npm check test inventory changed; update scaffold selection before splitting it");
  }
  const files = commands.flatMap((parts) => parts.slice(2));
  if (new Set(files).size !== files.length || HISTORICAL_TESTS.some((file) => !files.includes(file))) {
    throw new Error("npm check inventory is duplicated or missing a historical suite");
  }
  const selected = commands.map((parts) => parts.slice(2).filter((file) => HISTORICAL_TESTS.includes(file) === (mode === "history"))).filter((group) => group.length);
  if (!selected.length) throw new Error("empty scaffold selection");
  return selected;
}

export function scaffoldTestFiles(command, mode) { return scaffoldTestGroups(command, mode).flat(); }

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const command = JSON.parse(readFileSync("package.json", "utf8")).scripts.check;
  const mode = process.argv[2]?.replace(/^--/, "");
  const groups = scaffoldTestGroups(command, mode);
  console.log(`${mode}: ${groups.flat().length} complete test files from npm run check`);
  // Preserve the original && barrier: matrix-removed-sources mutates scratch copies of source
  // inputs and must retain its dedicated serial invocation after the parallel suite completes.
  for (const files of groups) {
    const child = spawnSync(process.execPath, ["--test", ...files], { stdio: "inherit", env: process.env });
    if (child.error) throw child.error;
    if (child.status !== 0) { process.exitCode = child.status ?? 1; break; }
  }
}
