import assert from "node:assert/strict";
import test from "node:test";

import { findRustCacheBinViolations } from "./rust-cache-bin.mjs";

const SHA = "e18b497796c12c097a38f9edb9d0641fb99eee32";
const USES = `      - uses: Swatinem/rust-cache@${SHA} # v2 (v2.9.1+1)`;

function workflow(...jobLines) {
  return ["jobs:", ...jobLines].join("\n");
}

test("an ephemeral job may use rust-cache with the default cache-bin", () => {
  const wf = workflow("  build:", "    runs-on: ubuntu-latest", "    steps:", USES);
  assert.deepEqual(findRustCacheBinViolations(wf, "wf.yml"), []);
});

test("a self-hosted job with cache-bin false is accepted", () => {
  const wf = workflow(
    "  nax:",
    "    runs-on: [self-hosted, macOS, ARM64]",
    "    steps:",
    USES,
    '        with:',
    '          cache-bin: "false"',
  );
  assert.deepEqual(findRustCacheBinViolations(wf, "wf.yml"), []);
});

test("a self-hosted job with the DEFAULT cache-bin is rejected — the regression this guards", () => {
  const wf = workflow("  nax:", "    runs-on: [self-hosted, macOS, ARM64]", "    steps:", USES);
  const violations = findRustCacheBinViolations(wf, "wf.yml");
  assert.equal(violations.length, 1);
  assert.equal(violations[0].job, "nax");
  assert.match(violations[0].reason, /without 'cache-bin: "false"'/);
});

test("a self-hosted job explicitly setting cache-bin true is rejected", () => {
  const wf = workflow(
    "  nax:",
    "    runs-on: [self-hosted, macOS, ARM64]",
    "    steps:",
    USES,
    "        with:",
    '          cache-bin: "true"',
  );
  const violations = findRustCacheBinViolations(wf, "wf.yml");
  assert.equal(violations.length, 1);
  assert.match(violations[0].reason, /must be "false"/);
});

test("runs-on as a block sequence is understood", () => {
  const wf = workflow(
    "  nax:",
    "    runs-on:",
    "      - self-hosted",
    "      - macOS",
    "    steps:",
    USES,
  );
  assert.equal(findRustCacheBinViolations(wf, "wf.yml").length, 1);
});

test("a plain self-hosted scalar runs-on is understood", () => {
  const wf = workflow("  nax:", "    runs-on: self-hosted", "    steps:", USES);
  assert.equal(findRustCacheBinViolations(wf, "wf.yml").length, 1);
});

test("a job using rust-cache with no runs-on at all is rejected rather than assumed ephemeral", () => {
  const wf = workflow("  mystery:", "    steps:", USES);
  const violations = findRustCacheBinViolations(wf, "wf.yml");
  assert.equal(violations.length, 1);
  assert.match(violations[0].reason, /declares no 'runs-on'/);
});

test("self-hosted jobs that do not use rust-cache are ignored", () => {
  const wf = workflow(
    "  package-windows:",
    "    runs-on: [self-hosted, windows]",
    "    steps:",
    "      - run: cargo build",
  );
  assert.deepEqual(findRustCacheBinViolations(wf, "wf.yml"), []);
});

test("cache-bin on a LATER step does not satisfy an earlier rust-cache step", () => {
  const wf = workflow(
    "  nax:",
    "    runs-on: [self-hosted, macOS]",
    "    steps:",
    USES,
    "      - uses: actions/checkout@" + SHA + " # v5",
    "        with:",
    '          cache-bin: "false"',
  );
  assert.equal(findRustCacheBinViolations(wf, "wf.yml").length, 1);
});

test("each offending step in a multi-step job is reported separately", () => {
  const wf = workflow(
    "  nax:",
    "    runs-on: [self-hosted, macOS]",
    "    steps:",
    USES,
    "      - run: echo between",
    USES,
  );
  assert.equal(findRustCacheBinViolations(wf, "wf.yml").length, 2);
});

test("the real repository workflows pass", async () => {
  const { readdir, readFile } = await import("node:fs/promises");
  const dir = ".github/workflows";
  const names = (await readdir(dir)).filter((name) => /\.ya?ml$/.test(name));
  assert.ok(names.length > 0, "expected workflows to scan");
  const found = (
    await Promise.all(
      names.map(async (name) =>
        findRustCacheBinViolations(await readFile(`${dir}/${name}`, "utf8"), `${dir}/${name}`),
      ),
    )
  ).flat();
  assert.deepEqual(found, []);
});
