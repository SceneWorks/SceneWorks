import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { SOURCE_PATHS as MATRIX_SOURCE_PATHS } from "./generate-memory-matrix.mjs";
import { PACKAGED_SOURCES_PATH } from "./extract-memory-anchors.mjs";
import { nodeTestsReading, selectSteps } from "./preflight.mjs";
import { SOURCE_PATHS as STALE_LANE_SOURCE_PATHS } from "./stale-lane-report.mjs";

const FULL_SUITE = "scaffold node suite (npm run check)";
const testFilesRun = (steps) =>
  steps.filter((step) => step.command?.[0] === "node" && step.command[1] === "--test").map((step) => step.command.slice(2));

// The PR #3066 shape: a Rust route source the matrix fingerprints changes, nothing under scripts/,
// config/ or docs/generated/ does, and the checked-in matrix goes stale. The freshness assertion
// (the shipped memory-matrix.md equals the renderer) lives in generate-memory-matrix.test.mjs.
test("a diff touching only a memory-matrix SOURCE_PATHS Rust file selects the matrix freshness test", () => {
  const steps = selectSteps([MATRIX_SOURCE_PATHS.imageRouting], false);
  assert.ok(!steps.some((step) => step.name === FULL_SUITE), "a Rust-only diff does not run the whole node suite");
  assert.ok(
    testFilesRun(steps).flat().includes("scripts/generate-memory-matrix.test.mjs"),
    "the matrix freshness test must run when a fingerprinted Rust source changes",
  );
});

test("every memory-matrix SOURCE_PATHS entry selects the matrix test or the whole suite", () => {
  // Derived from the generator's own map, so a source added there is asserted here with no edit.
  for (const [name, relative] of Object.entries(MATRIX_SOURCE_PATHS)) {
    const steps = selectSteps([relative], false);
    const covered =
      steps.some((step) => step.name === FULL_SUITE) ||
      testFilesRun(steps).flat().includes("scripts/generate-memory-matrix.test.mjs");
    assert.ok(covered, `SOURCE_PATHS.${name} (${relative}) changes the matrix but selects no matrix check`);
  }
});

test("the other source-reading generators' Rust inputs select their own tests", () => {
  for (const relative of Object.values(STALE_LANE_SOURCE_PATHS).filter((file) => file.endsWith(".rs"))) {
    assert.ok(
      testFilesRun(selectSteps([relative], false)).flat().includes("scripts/stale-lane-report.test.mjs"),
      `${relative} is a stale-lane-report source`,
    );
  }
  assert.ok(
    testFilesRun(selectSteps([PACKAGED_SOURCES_PATH], false)).flat().includes("scripts/extract-memory-anchors.test.mjs"),
    `${PACKAGED_SOURCES_PATH} is the anchor extractor's packaged-source list`,
  );
});

test("the serial npm run check segment stays a separate node --test run", () => {
  const runs = testFilesRun(selectSteps([MATRIX_SOURCE_PATHS.imageRouting], false));
  const serial = runs.find((files) => files.includes("scripts/matrix-removed-sources.test.mjs"));
  assert.deepEqual(serial, ["scripts/matrix-removed-sources.test.mjs"], "the disk-mutating probe never shares a parallel pool");
});

test("a scaffold-tree diff runs the whole suite once and no targeted subset", () => {
  const steps = selectSteps(["scripts/preflight.mjs", MATRIX_SOURCE_PATHS.imageRouting], false);
  assert.equal(steps.filter((step) => step.name === FULL_SUITE).length, 1);
  assert.deepEqual(testFilesRun(steps), []);
});

function fixtureRepo() {
  const root = mkdtempSync(path.join(os.tmpdir(), "sceneworks-preflight-test-"));
  const write = (relative, body) => {
    mkdirSync(path.dirname(path.join(root, relative)), { recursive: true });
    writeFileSync(path.join(root, relative), body);
  };
  write(
    "package.json",
    JSON.stringify({
      scripts: { check: "node --test scripts/a.test.mjs scripts/b.test.mjs && node --test scripts/serial.test.mjs" },
    }),
  );
  // a.test reaches the fingerprinted file only through a generator it imports, two hops away.
  write("scripts/a.test.mjs", 'import { buildThing } from "./gen.mjs";\n');
  write("scripts/gen.mjs", 'import { SOURCE_PATHS } from "./lib/sources.mjs";\nexport const buildThing = () => SOURCE_PATHS;\n');
  write(
    "scripts/lib/sources.mjs",
    'export const SOURCE_PATHS = Object.freeze({ route: "crates/w/src/route.rs", gone: "crates/w/src/deleted.rs" });\n',
  );
  // b.test names its input relative to itself, the way `new URL("../x", import.meta.url)` does.
  write("scripts/b.test.mjs", 'const url = new URL("../crates/w/src/guard.rs", import.meta.url);\n');
  write("scripts/serial.test.mjs", 'import { buildThing } from "./gen.mjs";\n');
  write("crates/w/src/route.rs", "pub fn route() {}\n");
  write("crates/w/src/guard.rs", "pub fn guard() {}\n");
  write("crates/w/src/unrelated.rs", "pub fn unrelated() {}\n");
  return root;
}

test("selection follows imports transitively and keeps each segment's own run", (t) => {
  const root = fixtureRepo();
  t.after(() => rmSync(root, { recursive: true, force: true }));
  assert.deepEqual(nodeTestsReading(["crates/w/src/route.rs"], root), [["scripts/a.test.mjs"], ["scripts/serial.test.mjs"]]);
  assert.deepEqual(nodeTestsReading(["crates/w/src/guard.rs"], root), [["scripts/b.test.mjs"]]);
});

test("an unrelated Rust file selects no node-suite test", (t) => {
  const root = fixtureRepo();
  t.after(() => rmSync(root, { recursive: true, force: true }));
  assert.deepEqual(nodeTestsReading(["crates/w/src/unrelated.rs"], root), []);
  assert.deepEqual(testFilesRun(selectSteps(["crates/w/src/unrelated.rs"], false, root)), []);
  // A literal naming a file that does not exist is not a read (and deleting it is a Rust-only change).
  assert.deepEqual(nodeTestsReading(["crates/w/src/deleted.rs"], root), []);
});
