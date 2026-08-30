#!/usr/bin/env node

// Deterministic source closure for the StarVector permanent-pin terminal candidate.
// The catalog itself is intentionally excluded: embedding its own byte hash would be recursive.

import { createHash } from "node:crypto";
import { lstat, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";

export const PRODUCTION_CLOSURE_PATHS = Object.freeze([
  ".github/workflows/starvector-terminal.yml",
  "Cargo.lock",
  "Cargo.toml",
  "apps/rust-api/src/main.rs",
  "crates/sceneworks-worker/Cargo.toml",
  "crates/sceneworks-worker/src/bin/starvector_terminal_lease.rs",
  "crates/sceneworks-worker/src/bin/starvector_terminal_sanitize.rs",
  "crates/sceneworks-worker/src/engines.rs",
  "crates/sceneworks-worker/src/inference_runtime.rs",
  "crates/sceneworks-worker/src/lib.rs",
  "crates/sceneworks-worker/src/model_jobs.rs",
  "crates/sceneworks-worker/src/prompt_refine_jobs.rs",
  "crates/sceneworks-worker/src/refine_model_cache.rs",
  "crates/sceneworks-worker/src/vector_admission.rs",
  "crates/sceneworks-worker/src/vector_jobs.rs",
  "package.json",
  "packages/schemas/model-manifest.schema.json",
  "release/starvector-terminal-campaign-v1.json",
  "release/starvector-terminal-metrics-lock-v1.json",
  "scripts/starvector-production-closure.mjs",
  "scripts/starvector-terminal-assets.mjs",
  "scripts/starvector-terminal-campaign.mjs",
  "scripts/starvector-terminal-case-bundle.mjs",
  "scripts/starvector-terminal-cli.mjs",
  "scripts/starvector-terminal-metrics.py",
  "scripts/starvector-terminal-producer.mjs",
  "scripts/starvector-terminal-product-service.mjs",
  "scripts/starvector-terminal-route.mjs",
]);

const SHA256 = /^[0-9a-f]{64}$/;

function fail(message) {
  throw new Error(`StarVector production closure: ${message}`);
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

export function normalizeClosurePath(input) {
  if (typeof input !== "string" || input.length === 0 || input.includes("\0")) {
    fail("entry path must be a non-empty UTF-8 string");
  }
  if (input.includes("\\")) fail(`entry path must use POSIX separators: ${input}`);
  if (path.posix.isAbsolute(input)) fail(`entry path must be relative: ${input}`);
  const segments = input.split("/");
  if (segments.some((segment) => segment === "" || segment === "." || segment === "..")) {
    fail(`entry path is not normalized: ${input}`);
  }
  const normalized = path.posix.normalize(input);
  if (normalized !== input || normalized === "config/manifests/builtin.models.jsonc") {
    fail(`entry path is not an allowed normalized source path: ${input}`);
  }
  return normalized;
}

export function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort(compareUtf8).map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

export function closureSha256(entries) {
  return sha256(stableJson({ entries }));
}

export async function buildProductionClosure({ root = process.cwd(), paths = PRODUCTION_CLOSURE_PATHS } = {}) {
  const seen = new Set();
  const normalized = paths.map((input) => {
    const entryPath = normalizeClosurePath(input);
    if (seen.has(entryPath)) fail(`duplicate entry path: ${entryPath}`);
    seen.add(entryPath);
    return entryPath;
  }).sort(compareUtf8);

  const entries = [];
  for (const entryPath of normalized) {
    const absolute = path.join(root, ...entryPath.split("/"));
    const info = await lstat(absolute).catch((error) => fail(`cannot stat ${entryPath}: ${error.message}`));
    if (info.isSymbolicLink()) fail(`symlinks are forbidden: ${entryPath}`);
    if (!info.isFile()) fail(`entry is not a regular file: ${entryPath}`);
    const bytes = await readFile(absolute);
    entries.push({ path: entryPath, byteSize: bytes.byteLength, sha256: sha256(bytes) });
  }
  return { schemaVersion: 1, sha256: closureSha256(entries), entries };
}

export function validateProductionClosureShape(closure) {
  if (!closure || typeof closure !== "object" || Array.isArray(closure)) fail("closure must be an object");
  const keys = Object.keys(closure).sort(compareUtf8);
  if (JSON.stringify(keys) !== JSON.stringify(["entries", "schemaVersion", "sha256"])) fail("closure keys are not exact");
  if (closure.schemaVersion !== 1 || !SHA256.test(closure.sha256) || !Array.isArray(closure.entries) || closure.entries.length === 0) {
    fail("closure header is malformed");
  }
  let previous = null;
  const seen = new Set();
  for (const entry of closure.entries) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) fail("entry must be an object");
    if (JSON.stringify(Object.keys(entry).sort(compareUtf8)) !== JSON.stringify(["byteSize", "path", "sha256"])) fail("entry keys are not exact");
    const entryPath = normalizeClosurePath(entry.path);
    if (!Number.isSafeInteger(entry.byteSize) || entry.byteSize < 0 || !SHA256.test(entry.sha256)) fail(`entry is malformed: ${entryPath}`);
    if (seen.has(entryPath)) fail(`duplicate entry path: ${entryPath}`);
    if (previous !== null && compareUtf8(previous, entryPath) >= 0) fail("entries are not strictly UTF-8 byte sorted");
    seen.add(entryPath);
    previous = entryPath;
  }
  if (closureSha256(closure.entries) !== closure.sha256) fail("aggregate sha256 does not match canonical {entries}");
  return closure;
}

export async function checkProductionClosure(closure, options = {}) {
  validateProductionClosureShape(closure);
  const live = await buildProductionClosure(options);
  if (stableJson(closure) !== stableJson(live)) fail("recorded closure differs from the current source tree");
  return live;
}

export async function checkManifestProductionClosure({
  root = process.cwd(),
  manifestPath = "config/manifests/builtin.models.jsonc",
  paths = PRODUCTION_CLOSURE_PATHS,
} = {}) {
  const manifest = JSON.parse(stripJsoncComments(await readFile(path.resolve(root, manifestPath), "utf8")));
  const model = manifest.models?.find(({ id }) => id === "starvector_8b");
  if (!model) fail("starvector_8b is missing from the builtin catalog");
  const candidate = model.vector?.deviceAdmission?.terminalCandidate;
  if (!candidate || typeof candidate !== "object") fail("starvector_8b terminalCandidate is missing");
  if (candidate.productionClosure === null) {
    for (const backend of ["mlx", "candle"]) {
      const provider = model.vector?.providers?.[backend];
      if (provider?.available !== false || provider.reason !== "pending_terminal_candidate") {
        fail(`a pending production closure requires fail-closed ${backend} availability`);
      }
    }
    return null;
  }
  return checkProductionClosure(candidate.productionClosure, { root, paths });
}

async function main(argv) {
  const [command, output] = argv;
  if (!command || !["render", "write", "check", "check-manifest"].includes(command)) {
    fail("usage: starvector-production-closure.mjs <render|write|check|check-manifest> [path]");
  }
  if (command === "check-manifest") {
    await checkManifestProductionClosure({ manifestPath: output ?? "config/manifests/builtin.models.jsonc" });
    return;
  }
  if (command === "render") {
    process.stdout.write(`${JSON.stringify(await buildProductionClosure(), null, 2)}\n`);
    return;
  }
  if (!output) fail(`${command} requires a closure JSON path`);
  if (command === "write") {
    const closure = await buildProductionClosure();
    await writeFile(output, `${JSON.stringify(closure, null, 2)}\n`, { flag: "w" });
    return;
  }
  const closure = JSON.parse(await readFile(output, "utf8"));
  await checkProductionClosure(closure);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  main(process.argv.slice(2)).catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
