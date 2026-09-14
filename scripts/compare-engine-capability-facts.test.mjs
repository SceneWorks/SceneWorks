import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { compareEngineCapabilityFacts } from "./compare-engine-capability-facts.mjs";

const REVISION_A = "a".repeat(40);
const REVISION_B = "b".repeat(40);
const CHECKER = join(dirname(fileURLToPath(import.meta.url)), "compare-engine-capability-facts.mjs");

async function withFacts(t, checkedIn, fresh) {
  const directory = await mkdtemp(join(tmpdir(), "sceneworks-capability-facts-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const checkedInPath = join(directory, "checked-in.json");
  const freshPath = join(directory, "fresh.json");
  await Promise.all([
    writeFile(checkedInPath, JSON.stringify(checkedIn)),
    writeFile(freshPath, JSON.stringify(fresh)),
  ]);
  return { checkedInPath, freshPath };
}

function facts(revision, supportsPreview = true) {
  return {
    backend: "candle",
    generatedFrom: {
      inferenceRevision: revision,
      dumper: "cargo run -p sceneworks-worker --bin dump-engine-capabilities",
    },
    engines: [{ id: "boogu_image", supportsPreview }],
  };
}

test("accepts a revision-only difference", async (t) => {
  const paths = await withFacts(t, facts(REVISION_A), facts(REVISION_B));
  const result = await compareEngineCapabilityFacts(paths.checkedInPath, paths.freshPath);
  assert.equal(result.matches, true);
  assert.equal(result.checkedInRevision, REVISION_A);
  assert.equal(result.freshRevision, REVISION_B);
});

test("rejects capability drift even when revisions differ", async (t) => {
  const paths = await withFacts(t, facts(REVISION_A), facts(REVISION_B, false));
  const result = await compareEngineCapabilityFacts(paths.checkedInPath, paths.freshPath);
  assert.equal(result.matches, false);
});

test("CLI reports capability drift as a failed check", async (t) => {
  const paths = await withFacts(t, facts(REVISION_A), facts(REVISION_B, false));
  const result = spawnSync(process.execPath, [CHECKER, paths.checkedInPath, paths.freshPath], {
    encoding: "utf8",
  });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /capability facts differ beyond generatedFrom\.inferenceRevision/);
});

test("rejects drift in other provenance fields", async (t) => {
  const checkedIn = facts(REVISION_A);
  const fresh = facts(REVISION_B);
  fresh.generatedFrom.dumper = "different dumper";
  const paths = await withFacts(t, checkedIn, fresh);
  const result = await compareEngineCapabilityFacts(paths.checkedInPath, paths.freshPath);
  assert.equal(result.matches, false);
});

test("rejects a missing or malformed revision instead of hiding it", async (t) => {
  const missing = facts(REVISION_A);
  delete missing.generatedFrom.inferenceRevision;
  const paths = await withFacts(t, missing, facts(REVISION_B));
  await assert.rejects(
    compareEngineCapabilityFacts(paths.checkedInPath, paths.freshPath),
    /inferenceRevision must be a 40-character lowercase SHA/,
  );
});
