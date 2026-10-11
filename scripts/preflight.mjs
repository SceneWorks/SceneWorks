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
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

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

// The parity-scaffold job points this at a checkout holding the pinned starvector contract commit;
// locally, any inference clone that contains it works. Without it three tests fail on a stale path.
const STARVECTOR_CONTRACT_REV = "0b084cbf46b5b4f4f561305ef116268b57eecf7f";

function git(...args) {
  const res = spawnSync("git", args, { cwd: repoRoot, encoding: "utf8" });
  if (res.status !== 0) throw new Error(`git ${args.join(" ")}: ${res.stderr.trim()}`);
  return res.stdout.trim();
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

function selectSteps(files, all) {
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

process.exit(main());
