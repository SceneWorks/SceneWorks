#!/usr/bin/env node
// Run, locally, the CI checks that a branch's changes can trip — before opening the PR, so CI
// confirms a green branch instead of discovering the failure. Selection keys off the files that
// differ from the merge base with the PR target (committed, uncommitted and untracked).
//
//   npm run preflight                                  # target origin/main
//   npm run preflight -- --base origin/feature/sc-NNNN # story PR into an epic branch
//   npm run preflight -- --all                         # every step, regardless of the diff
//   npm run preflight -- --dry-run                     # print the selected steps and exit
//   node scripts/preflight.mjs --verify [<sha>]        # exit 0 iff <sha> (default HEAD) passed
//
// Every selected step runs even after one fails, so a single pass reports every failure. A pass on
// a clean tree stamps the commit under <git-common-dir>/sceneworks-preflight/, shared by every
// worktree; `--verify` reads that stamp. The pre-push git hook and the Claude Code `gh pr create`
// hook both require it before anything that starts a CI run.
// Not covered here (CI only): the Windows CUDA lane's capability dump and smokes, the Docker
// runtime smokes, the pytest e2e/parity harnesses, and on macOS the Dockerized neither build.

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const isMac = process.platform === "darwin";

const RUST = /(^|\/)([^/]+\.rs|Cargo\.(toml|lock)|rustfmt\.toml)$|^\.cargo\/.*\.toml$/;
const WEB = /^(apps\/web\/|config\/|packages\/)/;
// Non-Rust inputs of the tier-integrity audit (same list as scripts/git-hooks/pre-push).
const TIER_INTEGRITY =
  /^(config\/manifests\/builtin\.models\.jsonc|config\/tier-integrity\.jsonc|packages\/schemas\/tier-integrity\.schema\.json|scripts\/(check-tier-integrity|tier-integrity-measurement-receipts)\.mjs|scripts\/lib\/jsonc\.mjs|docs\/generated\/tier-integrity\.(json|md))$/;
const CAPABILITIES = /^config\/engine-capabilities\//;
const PYTEST = /^(tests\/|config\/|scripts\/.*\.py$)/;
// The node suite reads manifests, generated docs, workflows and schemas as well as scripts.
const NODE = /^(scripts\/|config\/|docs\/generated\/|packages\/|\.github\/|package(-lock)?\.json$)/;

// The node suite also READS files outside NODE: the memory-matrix generator fingerprints Rust
// route/admission sources (`SOURCE_PATHS`) into the checked-in docs/generated/memory-matrix.*, the
// stale-lane report and closure/anchor tooling read the Cargo pin and the memory-adapter bins, and
// guard tests parse worker modules. An edit to one of those can red CI's parity-scaffold lane with
// no NODE file in the diff. Rather than mirror those lists here, derive them from the suite's own
// source text: every string literal in a node-suite test file, or in a local module it imports
// (transitively), that names an existing repo file is a file that test can read. A changed file
// outside NODE then selects exactly the test files that name it, run in their `npm run check`
// segment so the serial segment stays serial. A path a test only builds at runtime (a template or
// a `path.join` of fragments) is not seen; the generators' source lists are literal maps, so a new
// `SOURCE_PATHS` entry is covered with no change here.
const LITERAL = /(["'`])((?:\.{1,2}\/)*[A-Za-z0-9_@-][A-Za-z0-9_.@\/-]*)\1/g;
const IMPORT = /(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(["'])(\.{1,2}\/[^"']+\.mjs)\1/g;

function isRepoFile(relative, root) {
  if (relative.startsWith("..") || path.isAbsolute(relative)) return false;
  try {
    return statSync(path.join(root, relative)).isFile();
  } catch {
    return false;
  }
}

/** Repo-relative files named by a string literal in `file` (relative to the root or to `file`). */
function namedFiles(file, body, root) {
  const named = new Set();
  for (const [, , literal] of body.matchAll(LITERAL)) {
    const candidates = [literal.replace(/^(\.\/)+/, "")];
    if (literal.startsWith(".")) candidates.push(path.relative(root, path.resolve(root, path.dirname(file), literal)));
    for (const candidate of candidates) {
      const normalized = candidate.split(path.sep).join("/");
      if (isRepoFile(normalized, root)) named.add(normalized);
    }
  }
  return named;
}

/** `npm run check` as its `&&` segments, each the ordered list of test files one `node --test` runs. */
export function nodeSuiteSegments(root = repoRoot) {
  const script = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8")).scripts.check;
  return script
    .split("&&")
    .map((segment) => segment.trim().split(/\s+/))
    .filter((tokens) => tokens[0] === "node" && tokens[1] === "--test")
    .map((tokens) => tokens.filter((token) => token.endsWith(".test.mjs")));
}

/** Map each node-suite test file to every repo file it, or a local module it imports, names. */
export function nodeSuiteReads(root = repoRoot) {
  const cache = new Map();
  const closure = (file, seen) => {
    if (seen.has(file)) return;
    seen.add(file);
    let body;
    try {
      body = readFileSync(path.join(root, file), "utf8");
    } catch {
      return;
    }
    if (!cache.has(file)) cache.set(file, namedFiles(file, body, root));
    for (const [, , specifier] of body.matchAll(IMPORT)) {
      closure(path.relative(root, path.resolve(root, path.dirname(file), specifier)).split(path.sep).join("/"), seen);
    }
  };
  const reads = new Map();
  for (const testFile of nodeSuiteSegments(root).flat()) {
    const seen = new Set();
    closure(testFile, seen);
    reads.set(testFile, new Set([...seen].flatMap((module) => [...(cache.get(module) ?? [])])));
  }
  return reads;
}

/** Per `npm run check` segment, the test files that read one of `files` (files outside NODE only). */
export function nodeTestsReading(files, root = repoRoot) {
  const outside = files.filter((file) => !NODE.test(file));
  if (outside.length === 0) return [];
  const reads = nodeSuiteReads(root);
  return nodeSuiteSegments(root)
    .map((segment) => segment.filter((testFile) => outside.some((file) => reads.get(testFile)?.has(file))))
    .filter((segment) => segment.length > 0);
}

// The parity-scaffold job points this at a checkout holding the pinned starvector contract commit;
// locally, any inference clone that contains it works. Without it three tests fail on a stale path.
const STARVECTOR_CONTRACT_REV = "0b084cbf46b5b4f4f561305ef116268b57eecf7f";

export function gitOutput(args, { cwd = repoRoot, run = spawnSync } = {}) {
  // Preserved local build/evidence trees can exceed spawnSync's 1 MiB default listing buffer.
  // Bound memory without silently dropping paths from preflight's changed-file selection.
  const res = run("git", args, { cwd, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  if (res.error || res.status !== 0) {
    const detail = res.error
      ? `${res.error.code ?? "spawn error"}: ${res.error.message}`
      : `exit ${res.status}${res.signal ? ` (${res.signal})` : ""}`;
    throw new Error(`git ${args.join(" ")}: ${detail}${res.stderr?.trim() ? `: ${res.stderr.trim()}` : ""}`);
  }
  return res.stdout.trim();
}

function git(...args) {
  return gitOutput(args);
}

function stampPath(sha) {
  return path.resolve(repoRoot, git("rev-parse", "--git-common-dir"), "sceneworks-preflight", sha);
}

function verify(rev) {
  const sha = git("rev-parse", "--verify", `${rev}^{commit}`);
  const stamp = stampPath(sha);
  if (existsSync(stamp)) {
    console.log(`preflight passed for ${sha.slice(0, 9)}:\n${readFileSync(stamp, "utf8")}`);
    return 0;
  }
  console.error(`No passing preflight for ${sha.slice(0, 9)}. Commit everything, then run:`);
  console.error("  npm run preflight -- --base <PR target, e.g. origin/feature/sc-NNNN or origin/main>");
  return 1;
}

function changedFiles(base) {
  const mergeBase = git("merge-base", "HEAD", base);
  const tracked = git("diff", "--name-only", mergeBase);
  const untracked = git("ls-files", "--others", "--exclude-standard");
  return [...new Set(`${tracked}\n${untracked}`.split("\n").filter(Boolean))];
}

function nodeSuiteEnv() {
  if (process.env.STARVECTOR_TERMINAL_INFERENCE_TEST_ROOT) return {};
  const clone = process.env.SCENEWORKS_INFERENCE_REPO ?? path.resolve(os.homedir(), "Repos/inference");
  const probe = spawnSync("git", ["-C", clone, "cat-file", "-e", `${STARVECTOR_CONTRACT_REV}^{commit}`]);
  return probe.status === 0 ? { STARVECTOR_TERMINAL_INFERENCE_TEST_ROOT: clone } : {};
}

// Mirrors macos-mlx.yml "Verify capabilities.mlx.json content against a fresh dump".
function capabilityDumpCheck() {
  const scratch = mkdtempSync(path.join(os.tmpdir(), "sceneworks-capabilities-"));
  try {
    const dump = run(["cargo", "run", "-q", "-p", "sceneworks-worker", "--bin", "dump-engine-capabilities", "--", scratch]);
    if (dump !== 0) return dump;
    let status = 0;
    for (const file of ["capabilities.mlx.json", "runtime/capabilities.mlx.json", "audio/capabilities.candle.json"]) {
      const compare = run([
        "node",
        "scripts/compare-engine-capability-facts.mjs",
        `config/engine-capabilities/${file}`,
        path.join(scratch, file),
      ]);
      if (compare !== 0) {
        console.error(`config/engine-capabilities/${file} differs from a fresh dump: re-dump with`);
        console.error("  cargo run -p sceneworks-worker --bin dump-engine-capabilities");
        console.error("then (cd apps/web && npm run gen:preview-support).");
        status = compare;
      }
    }
    return status;
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

// check.yml parity-rust installs exactly these; Homebrew's python3 refuses a system-wide pip install
// (PEP 668), so they live in a venv shared by every worktree and rebuilt when the pins change.
const PYTEST_DEPS = ["pytest==9.0.2", "httpx==0.28.1", "jsonschema==4.25.1"];

function workerPytest() {
  const venv = path.resolve(repoRoot, git("rev-parse", "--git-common-dir"), "sceneworks-preflight", "venv");
  const python = path.join(venv, "bin", "python");
  const marker = path.join(venv, "preflight-deps.txt");
  const wanted = PYTEST_DEPS.join("\n");
  if (!existsSync(marker) || readFileSync(marker, "utf8") !== wanted) {
    rmSync(venv, { recursive: true, force: true });
    const created =
      run(["python3", "-m", "venv", venv]) || run([python, "-m", "pip", "install", "-q", "--disable-pip-version-check", ...PYTEST_DEPS]);
    if (created !== 0) return created;
    writeFileSync(marker, wanted);
  }
  return run([python, "-m", "pytest", "-q", "tests/", "-m", "not e2e and not parity", "--strict-markers"]);
}

function run(command, env = {}) {
  const res = spawnSync(command[0], command.slice(1), {
    cwd: repoRoot,
    stdio: "inherit",
    env: { ...process.env, ...env },
  });
  return res.error ? 1 : res.status;
}

export function selectSteps(files, all, root = repoRoot) {
  const touches = (pattern) => all || files.some((file) => pattern.test(file));
  const steps = [];
  if (touches(RUST)) {
    // check.yml parity-rust + candle; macos-mlx.yml macos-checks.
    steps.push({ name: "rust:check (tier docs, fmt, clippy, test, build)", command: ["npm", "run", "rust:check"] });
    // On macOS the neither build needs a Linux Docker container (30-42 GB volume per worktree);
    // the dev Mac does not run containers, so that lane is left to CI's parity job.
    if (!isMac) steps.push({ name: "neither build (Linux, no candle)", command: ["npm", "run", "rust:check:neither"] });
    steps.push({ name: "candle build (not-macOS + candle)", command: ["npm", "run", "rust:check:candle", "--", "--allow-skip"] });
    if (isMac) {
      steps.push({
        name: "MLX memory adapter clippy",
        command: ["cargo", "clippy", "-p", "sceneworks-memory-adapter", "--features", "mlx", "--all-targets", "--", "-D", "warnings"],
      });
      steps.push({
        name: "MLX memory adapter tests",
        command: ["cargo", "test", "-p", "sceneworks-memory-adapter", "--features", "mlx"],
      });
    }
  }
  if (!touches(RUST) && touches(TIER_INTEGRITY)) {
    steps.push({ name: "tier-integrity audit", command: ["npm", "run", "check:rust-derived-docs"] });
  }
  if (isMac && (touches(RUST) || touches(CAPABILITIES))) {
    steps.push({ name: "capabilities.mlx.json vs fresh dump", fn: capabilityDumpCheck });
  }
  if (touches(WEB)) {
    steps.push({ name: "web lint", command: ["npm", "--prefix", "apps/web", "run", "lint"] });
    steps.push({ name: "web tests", command: ["npm", "--prefix", "apps/web", "run", "test"] });
    steps.push({ name: "web build", command: ["npm", "--prefix", "apps/web", "run", "build"] });
  }
  if (touches(NODE)) {
    // Several suites are wall-clock sensitive (watchdog ceilings, port races) and flake under the
    // parallel runner on a loaded Mac, so one re-run is allowed and reported as such.
    steps.push({ name: "scaffold node suite (npm run check)", command: ["npm", "run", "check"], env: nodeSuiteEnv(), retry: true });
  } else {
    for (const segment of nodeTestsReading(files, root)) {
      steps.push({
        name: `node-suite tests reading changed non-scaffold sources (${segment.join(", ")})`,
        command: ["node", "--test", ...segment],
        env: nodeSuiteEnv(),
        retry: true,
      });
    }
  }
  if (touches(PYTEST)) {
    steps.push({ name: "worker pytest (not e2e, not parity)", fn: workerPytest });
  }
  return steps;
}

function main() {
  const args = process.argv.slice(2);
  const verifyIndex = args.indexOf("--verify");
  if (verifyIndex >= 0) return verify(args[verifyIndex + 1] ?? "HEAD");
  const all = args.includes("--all");
  const dryRun = args.includes("--dry-run");
  const baseIndex = args.indexOf("--base");
  const base = baseIndex >= 0 ? args[baseIndex + 1] : "origin/main";
  if (!base) throw new Error("--base needs a ref, e.g. --base origin/feature/sc-NNNN");

  const files = all ? [] : changedFiles(base);
  const steps = selectSteps(files, all);
  console.log(`preflight: ${all ? "--all" : `${files.length} file(s) changed vs merge-base with ${base}`}`);
  for (const step of steps) console.log(`  - ${step.name}`);
  if (steps.length === 0) console.log("  (no CI-relevant changes)");
  if (dryRun) return 0;

  const results = [];
  for (const step of steps) {
    console.log(`\n=== preflight: ${step.name}`);
    const started = Date.now();
    let status = step.fn ? step.fn() : run(step.command, step.env);
    let retried = false;
    if (status !== 0 && step.retry) {
      console.log(`\n=== preflight: ${step.name} failed; re-running once`);
      status = run(step.command, step.env);
      retried = true;
    }
    const ok = status === 0;
    const label = ok ? (retried ? "PASS (on retry)" : "PASS") : "FAIL";
    results.push({ name: step.name, label, ok, seconds: Math.round((Date.now() - started) / 1000) });
  }

  const head = git("rev-parse", "HEAD");
  const dirty = git("status", "--porcelain") !== "";
  const passed = results.every((r) => r.ok);
  const summary = [
    `preflight summary @ ${head.slice(0, 9)}${dirty ? " (uncommitted changes)" : ""}, base ${base}${all ? ", --all" : ""}:`,
    ...results.map((r) => `  ${r.label}  ${r.name} (${r.seconds}s)`),
  ];
  if (results.length === 0) summary.push("  (no CI-relevant changes)");
  console.log(`\n${summary.join("\n")}`);
  if (passed && !dirty) {
    const stamp = stampPath(head);
    mkdirSync(path.dirname(stamp), { recursive: true });
    writeFileSync(stamp, `${summary.join("\n")}\n`);
    console.log(`stamped ${head.slice(0, 9)}: pushes to PR branches and gh pr create are unblocked.`);
  } else if (passed) {
    console.log("NOT stamped: the tree has uncommitted changes. Commit, then re-run preflight.");
  }
  return passed ? 0 : 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) process.exit(main());
