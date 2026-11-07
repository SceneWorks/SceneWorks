#!/usr/bin/env node
// Capture typed data, never infer provider contracts by parsing Rust source. The old fixture is
// immutable during comparisons: only this explicit, source-identified capture can replace it.
import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { renderBaseline } from "./baseline-json.mjs";

const root = fileURLToPath(new URL("../../", import.meta.url));
const git = (...args) => execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
const { values } = parseArgs({ options: { source: { type: "string" }, output: { type: "string" } } });
if (!values.source || !values.output) {
  throw new Error("Usage: capture-catalog-baseline.mjs --source <full old SHA> --output <fixture.json>");
}
const source = values.source;
if (!/^[a-f0-9]{40}$/.test(source) || git("rev-parse", "HEAD") !== source) {
  throw new Error("Capture must run in the pinned full-SHA old checkout before registry edits");
}
const paths = ["apps/web/src", "config/manifests", "config/engine-capabilities", "crates/sceneworks-core/src/jobs_store", "crates/sceneworks-core/src/training.rs", "apps/rust-api/src/models", "apps/rust-api/src/training.rs"];
if (git("diff", source, "--", ...paths)) throw new Error("Old product sources have uncommitted changes");
const metadata = JSON.parse(execFileSync("cargo", ["metadata", "--locked", "--no-deps", "--format-version", "1"], { cwd: root, encoding: "utf8" }));
const dependency = metadata.packages.flatMap((p) => p.dependencies).find((d) => d.name === "sceneworks-gen-core");
const dump = execFileSync("cargo", ["run", "--locked", "-q", "-p", "sceneworks-core", "--bin", "dump-ui-catalog"], { cwd: root, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
const result = {
  schemaVersion: 2,
  source: { repository: "SceneWorks/SceneWorks", revision: source, inferenceSource: dependency.source,
    nativeDump: "cargo run --locked -p sceneworks-core --bin dump-ui-catalog",
    trees: Object.fromEntries(paths.map((path) => [path, git("rev-parse", `${source}:${path}`)])),
    hostFixture: "All builtin artifacts and declared tiers installed; fixed project assets; no filesystem probing, fetch, weights or model execution. Native support, preview and decoder projections use the production core functions." },
  catalog: JSON.parse(dump),
};
writeFileSync(resolve(values.output), `${renderBaseline(result)}\n`);
console.log(`Captured ${result.catalog.models.length} old models at ${source}`);
