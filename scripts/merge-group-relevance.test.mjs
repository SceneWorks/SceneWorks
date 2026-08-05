import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  matchesGlob,
  parseWorkflowPaths,
  relevanceDecision,
} from "./merge-group-relevance.mjs";

async function source(path) {
  return readFile(new URL(`../${path}`, import.meta.url), "utf8");
}

test("parses the real macos-mlx anchor, comments and all", async () => {
  const paths = parseWorkflowPaths(await source(".github/workflows/macos-mlx.yml"), "mlx_paths");
  // The anchor is heavily interleaved with multi-paragraph rationale comments; every one of them
  // must be skipped rather than ending the block early.
  for (const watched of [
    "crates/**",
    "apps/rust-worker/**",
    "apps/rust-api/**",
    "config/manifests/**",
    "config/engine-capabilities/capabilities.mlx.json",
    "config/engine-capabilities/audio/capabilities.candle.json",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
    ".github/workflows/macos-mlx.yml",
  ]) {
    assert.ok(paths.includes(watched), `mlx_paths should carry ${watched}, got ${paths.join(", ")}`);
  }
  // The block must STOP at `pull_request:` — not run on and swallow the workflow_dispatch inputs.
  assert.ok(!paths.some((entry) => entry.includes("bf16")), paths.join(", "));
});

test("parses the real windows-candle anchor", async () => {
  const paths = parseWorkflowPaths(
    await source(".github/workflows/windows-candle.yml"),
    "candle_paths",
  );
  assert.ok(paths.includes("crates/**"));
  assert.ok(paths.includes("config/engine-capabilities/capabilities.candle.json"));
  // sc-17703 deliberately keeps apps/rust-worker/** OFF this lane; the parse must reflect the file,
  // not the sibling lane's list.
  assert.ok(!paths.includes("apps/rust-worker/**"), paths.join(", "));
});

test("the block ends at the next `on:` key", () => {
  const workflow = [
    "on:",
    "  push:",
    "    paths: &p",
    '      - "crates/**"',
    "      # a comment",
    "",
    '      - "Cargo.toml"',
    "  pull_request:",
    "    branches: [main]",
    '      - "not-a-path"',
  ].join("\n");
  assert.deepEqual(parseWorkflowPaths(workflow, "p"), ["crates/**", "Cargo.toml"]);
});

test("an unquoted or missing anchor is an error, never a silently short list", () => {
  assert.throws(
    () => parseWorkflowPaths(['    paths: &p', "      - crates/**"].join("\n"), "p"),
    /not a bare double-quoted string/,
  );
  assert.throws(() => parseWorkflowPaths("on:\n  push:\n", "p"), /declares no/);
  assert.throws(() => parseWorkflowPaths("    paths: &p\n  push:\n", "p"), /is empty/);
});

test("glob semantics match GitHub filter patterns", () => {
  assert.ok(matchesGlob("crates/**", "crates/sceneworks-core/tests/jobs_store.rs"));
  assert.ok(matchesGlob("crates/**", "crates/a"));
  assert.ok(!matchesGlob("crates/**", "apps/rust-api/src/main.rs"));
  assert.ok(matchesGlob("Cargo.toml", "Cargo.toml"));
  // A bare filename anchors — it must not match the same name nested in a subdirectory.
  assert.ok(!matchesGlob("Cargo.toml", "crates/sceneworks-core/Cargo.toml"));
  assert.ok(matchesGlob("config/engine-capabilities/capabilities.mlx.json", "config/engine-capabilities/capabilities.mlx.json"));
  assert.ok(!matchesGlob("config/engine-capabilities/capabilities.mlx.json", "config/engine-capabilities/capabilities.candle.json"));
});

test("a docs-only merge group skips the lane; a crates-only one runs it", async () => {
  const paths = parseWorkflowPaths(await source(".github/workflows/macos-mlx.yml"), "mlx_paths");

  const docs = relevanceDecision(paths, ["README.md", "docs/rust-mlx-build.md", "apps/web/src/App.jsx"]);
  assert.equal(docs.run, false);

  // The exact file today's semantic conflict landed in (e71b932f).
  const rust = relevanceDecision(paths, ["crates/sceneworks-core/tests/jobs_store.rs"]);
  assert.equal(rust.run, true);
  assert.match(rust.reason, /crates\/\*\*/);
});

test("an undeterminable diff runs the lane rather than skipping it", () => {
  const decision = relevanceDecision(["crates/**"], []);
  assert.equal(decision.run, true);
  assert.match(decision.reason, /conservatively/);
});
