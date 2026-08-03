import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { HARNESS_VERSION } from "./memory-calibration-harness.mjs";
import {
  assertCanonicalBundleMatches, certificationBundle,
} from "./sc-16170-certification.mjs";

const sourceSession = {
  id: `ims-${"1".repeat(20)}`,
  kind: "unit_test",
  command: "cargo test -p candle-gen-z-image",
  sourcePath: "docs/calibration/sc-16170/lifecycle.log",
  capturedAt: "2026-08-03T12:00:00Z",
  repositories: {
    sceneWorks: { revision: "a".repeat(40), dirty: false },
    inference: { revision: "b".repeat(40), dirty: false },
  },
  hardware: { probe: "fixture CUDA probe", memoryBytes: 1024 },
  stdoutSha256: "c".repeat(64),
  inputs: [],
  outputs: [],
  claims: ["lifecycle"],
  result: "passed",
};

test("SC-16170 comparison is key-order independent and preserves source sessions", () => {
  const expected = certificationBundle({ sourceSessions: [sourceSession] }, []);
  assert.deepEqual(expected.sourceSessions, [sourceSession]);

  const reordered = {
    records: [],
    sourceSessions: [{
      result: "passed",
      claims: ["lifecycle"],
      outputs: [],
      inputs: [],
      stdoutSha256: "c".repeat(64),
      hardware: { memoryBytes: 1024, probe: "fixture CUDA probe" },
      repositories: {
        inference: { dirty: false, revision: "b".repeat(40) },
        sceneWorks: { dirty: false, revision: "a".repeat(40) },
      },
      capturedAt: "2026-08-03T12:00:00Z",
      sourcePath: "docs/calibration/sc-16170/lifecycle.log",
      command: "cargo test -p candle-gen-z-image",
      kind: "unit_test",
      id: `ims-${"1".repeat(20)}`,
    }],
    harnessVersion: HARNESS_VERSION,
    schemaVersion: 4,
  };
  assert.doesNotThrow(() => assertCanonicalBundleMatches(reordered, expected));
});

test("SC-16170 comparison detects material source-session and record drift", () => {
  const expected = certificationBundle({ sourceSessions: [sourceSession] }, []);
  const missingSession = certificationBundle({ sourceSessions: [] }, []);
  assert.throws(
    () => assertCanonicalBundleMatches(missingSession, expected),
    /material SC-16170 drift/,
  );

  const checkedIn = JSON.parse(readFileSync(
    new URL("../docs/generated/memory-calibration-evidence.json", import.meta.url),
    "utf8",
  ));
  const materialRecord = structuredClone(checkedIn);
  assert.ok(materialRecord.records.length > 0, "fixture must exercise record drift");
  materialRecord.records.pop();
  assert.throws(
    () => assertCanonicalBundleMatches(materialRecord, checkedIn),
    /material SC-16170 drift/,
  );
});
