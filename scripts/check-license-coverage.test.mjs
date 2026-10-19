// sc-19751: prove the license coverage check REPORTS rather than gates.
//
// `check-license-coverage.mjs` runs its whole analysis at import time and exits, so there are no
// pure functions to import — these drive the real script as a subprocess. (The comparison here used
// to name `check-evidence-corpus.test.mjs`, which sc-22512 removed along with its gate.)
//
// The mutation used throughout is the one that actually bit us: pointing the audited inventory at a
// revision other than the Cargo pin, which is what EVERY inference pin bump did. Asserting on a
// finding that a clean tree does not already produce is deliberate — on a clean tree the check
// passes, so a test that only ran it there would pass with the feature reverted and prove nothing.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SCRIPT = path.join(ROOT, "scripts/check-license-coverage.mjs");
const AUDIT = path.join(ROOT, "config/inference-third-party-source.json");

// The checker derives its root from import.meta.url, NOT cwd. Copy its read-only packaging
// inputs and scripts into an owned fixture. Never mutate a tracked audit and hope finally can
// restore it: a Windows sharing error on that restore poisoned the following full-suite retry.
const CHECKER_INPUTS = [
  "scripts/check-license-coverage.mjs", "scripts/scan-inference-provenance.mjs",
  "config/manifests/builtin.models.jsonc", "config/inference-third-party-source.json",
  "config/inference-provenance-candidates.tsv", "config/inference-crate-prefixes.txt",
  "crates/sceneworks-worker/Cargo.toml", "Cargo.toml", "Cargo.lock", "package.json",
  "apps/desktop/licenses", "apps/web/src/data/bundledLicenses.js",
  "apps/desktop/tauri.conf.json", "apps/rust-api/src/lib.rs", "apps/desktop/package.json",
  "apps/desktop/scripts/build-sidecar.mjs", "apps/desktop/scripts/build-sidecar-platform.mjs",
];

function withPinSkew(run) {
  const liveAudit = readFileSync(AUDIT, "utf8");
  const fixture = mkdtempSync(path.join(os.tmpdir(), "license-coverage-fixture-"));
  try {
    for (const relative of CHECKER_INPUTS) {
      const target = path.join(fixture, relative);
      mkdirSync(path.dirname(target), { recursive: true });
      cpSync(path.join(ROOT, relative), target, { recursive: true });
    }
    const fixtureAudit = path.join(fixture, "config/inference-third-party-source.json");
    assert.notEqual(fixtureAudit, AUDIT, "mutation must never target the live checkout");
    assert.ok(path.relative(fixture, fixtureAudit).startsWith(`config${path.sep}`));
    const audit = JSON.parse(readFileSync(fixtureAudit, "utf8"));
    audit.inferenceRevision = "0".repeat(40);
    writeFileSync(fixtureAudit, `${JSON.stringify(audit, null, 2)}\n`);
    return run((args = []) => runChecker(args, path.join(fixture, "scripts/check-license-coverage.mjs")), fixture);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
    assert.equal(readFileSync(AUDIT, "utf8"), liveAudit, "isolated checker must leave the live audit byte-for-byte unchanged");
  }
}

// Findings are written to stderr and the summary to stdout, so both streams have to be captured —
// `execFileSync` returns stdout only, which made an earlier version of this test read a clean-tree
// summary and miss the report entirely.
function runChecker(args = [], script = SCRIPT) {
  const result = spawnSync(process.execPath, [script, ...args], { encoding: "utf8", cwd: ROOT });
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

test("pin-skew subprocess uses its isolated script root and never writes the live checkout", () => {
  const before = readFileSync(AUDIT, "utf8");
  withPinSkew((check, fixture) => {
    assert.notEqual(fixture, ROOT);
    assert.equal(JSON.parse(readFileSync(path.join(fixture, "config/inference-third-party-source.json"), "utf8")).inferenceRevision, "0".repeat(40));
    assert.equal(readFileSync(AUDIT, "utf8"), before);
    const { status, output } = check(["--strict"]);
    assert.equal(status, 1, output);
    assert.match(output, /Re-audit inference NOTICE/);
    assert.equal(readFileSync(AUDIT, "utf8"), before);
  });
  assert.equal(readFileSync(AUDIT, "utf8"), before);
});

test("a pin bump alone does not fail the build", () => {
  const { status, output } = withPinSkew((check) => check());
  assert.equal(status, 0, `an inference pin bump must not fail CI (sc-19751):\n${output}`);
  assert.match(output, /REPORT — not enforced/);
  assert.match(output, /Re-audit inference NOTICE/, "the finding must still be reported, not hidden");
  assert.match(output, /do NOT fail the build/);
});

test("--strict still gates, so the compliance pass has a lever", () => {
  const { status, output } = withPinSkew((check) => check(["--strict"]));
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
