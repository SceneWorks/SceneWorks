// sc-19751: prove the license coverage check REPORTS rather than gates.
//
// `check-license-coverage.mjs` runs its whole analysis at import time and exits, so unlike
// `check-evidence-corpus.test.mjs` there are no pure functions to import — these drive the real
// script as a subprocess.
//
// The mutation used throughout is the one that actually bit us: pointing the audited inventory at a
// revision other than the Cargo pin, which is what EVERY inference pin bump did. Asserting on a
// finding that a clean tree does not already produce is deliberate — on a clean tree the check
// passes, so a test that only ran it there would pass with the feature reverted and prove nothing.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SCRIPT = path.join(ROOT, "scripts/check-license-coverage.mjs");
const AUDIT = path.join(ROOT, "config/inference-third-party-source.json");

/** Run the checker with the audit temporarily pointed at a foreign revision. */
function withPinSkew(run) {
  const original = readFileSync(AUDIT, "utf8");
  try {
    const audit = JSON.parse(original);
    audit.inferenceRevision = "0".repeat(40);
    writeFileSync(AUDIT, `${JSON.stringify(audit, null, 2)}\n`);
    return run();
  } finally {
    writeFileSync(AUDIT, original);
  }
}

// Findings are written to stderr and the summary to stdout, so both streams have to be captured —
// `execFileSync` returns stdout only, which made an earlier version of this test read a clean-tree
// summary and miss the report entirely.
function runChecker(args = []) {
  const result = spawnSync("node", [SCRIPT, ...args], { encoding: "utf8" });
  if (result.error) throw result.error;
  return { status: result.status, output: `${result.stdout ?? ""}${result.stderr ?? ""}` };
}

test("the audit comment never restates the mutable inference pin", () => {
  const audit = JSON.parse(readFileSync(AUDIT, "utf8"));
  assert.doesNotMatch(
    audit._comment ?? "",
    /\b[0-9a-f]{40}\b/i,
    "inferenceRevision is the sole pin authority; reviewer prose must stay revision-neutral",
  );
});

test("a pin bump alone does not fail the build", () => {
  const { status, output } = withPinSkew(() => runChecker());
  assert.equal(status, 0, `an inference pin bump must not fail CI (sc-19751):\n${output}`);
  assert.match(output, /REPORT — not enforced/);
  assert.match(output, /Re-audit inference NOTICE/, "the finding must still be reported, not hidden");
  assert.match(output, /do NOT fail the build/);
});

test("--strict still gates, so the compliance pass has a lever", () => {
  const { status, output } = withPinSkew(() => runChecker(["--strict"]));
  assert.equal(status, 1, `--strict must still exit non-zero:\n${output}`);
  assert.match(output, /License coverage check FAILED/);
});

test("the analysis is unchanged — every rule still detects its mutation", () => {
  const { status, output } = runChecker(["--self-test"]);
  assert.equal(status, 0, output);
  assert.match(output, /self-test PASS/);
});

// The default run is REPORT-ONLY, and this test must not quietly re-impose the gate that
// sc-19751/sc-19758 deliberately dismantled. It used to assert `[license-coverage] OK` and the
// ABSENCE of the report banner — but any finding at all makes the script print the report banner
// instead of OK, so those two assertions turned every finding into a hard `npm run check` failure
// in the parity-scaffold job. That is exactly the gate the teardown removed, reintroduced through
// the back door of a test. All the default run owes anyone is: it ran, it said something, and it
// did not fail the build. `--strict` above is the deliberate lever for the compliance pass, and the
// pin-skew test above still proves findings are reported rather than hidden.
test("the default run reports and never fails the build, findings or not", () => {
  const { status, output } = runChecker();
  assert.equal(status, 0, output);
  assert.match(output, /\[license-coverage\]/);
});
