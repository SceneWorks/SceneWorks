#!/usr/bin/env node
// What disagrees between the engine registries, the manifest declarations and the route witnesses?
//
// WHY THIS EXISTS
// ---------------
// Michael, 2026-08-17: drop the gate entirely, lose the whole concept of waivers, keep the
// reconciliation as a report.
//
// The reconciliation used to end in `config/memory-contract-reconciliation-waivers.json` — every
// accepted mismatch listed by all eleven axes plus the provider's `selectorDigest`, with a bijection
// check that failed the build on any unwaived mismatch OR any waiver with no live mismatch. That
// ledger was pin-keyed, so an inference pin bump staled it wholesale: landing SC-18460 on the current
// epic head produced 253 unwaived mismatches and 382 stale waivers at once, and only 101 of those were
// the same coordinate with a rotated digest. Making it green would have meant authoring 152 new waivers
// with invented owner stories. The ledger, its schema, the `ownerStory`/`reason` fields and the
// bijection are deleted.
//
// What had value was the enumeration — it is how 152 engine surfaces with no manifest declaration were
// found at all — so that survives here. This REPORTS and always exits 0. It is deliberately not wired
// into CI. Recording is not enforcing; runtime catching is the chosen tradeoff. A human runs this,
// reads it, and decides what to declare or retire.
//
// Usage:
//   node scripts/report-memory-contract-reconciliation.mjs
//   node scripts/report-memory-contract-reconciliation.mjs --json
//   node scripts/report-memory-contract-reconciliation.mjs --leg engine_manifest

import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const asJson = argv.includes("--json");
const legAt = argv.indexOf("--leg");
const legFilter = legAt >= 0 ? argv[legAt + 1] : null;

// The reconciliation needs the same assembled inputs the matrix generator builds (cells, calibration
// plan, survey), so the report asks the generator for them rather than reassembling — one code path,
// one answer. `--emit-reconciliation` prints the reconciliation result as JSON and nothing else.
function loadReconciliation() {
  const raw = execFileSync(
    process.execPath,
    [path.join(ROOT, "scripts/generate-memory-matrix.mjs"), "--emit-reconciliation"],
    { cwd: ROOT, encoding: "utf8", maxBuffer: 256 * 1024 * 1024 },
  );
  return JSON.parse(raw);
}

let result;
try {
  result = loadReconciliation();
} catch (error) {
  // Even the report refuses to fail. If the inputs cannot be assembled at all, say so and exit 0 —
  // this script is a worklist, and an unavailable worklist is not a build failure.
  console.log("Memory-contract reconciliation report");
  console.log("  Could not assemble the reconciliation inputs on this branch:");
  console.log(`  ${(error?.stderr || error?.message || String(error)).toString().trim().split("\n").slice(-4).join("\n  ")}`);
  console.log("\nDone. Exit 0 always.");
  process.exit(0);
}

const findings = (result.findings ?? []).filter(
  (entry) => !legFilter || entry.leg === legFilter,
);

if (asJson) {
  console.log(JSON.stringify({ ...result, findings }, null, 2));
  process.exit(0);
}

const plural = (count, noun, plural = `${noun}s`) => `${count} ${count === 1 ? noun : plural}`;
const coordinate = (entry) =>
  [
    entry.backend,
    entry.provider,
    entry.modelId,
    entry.tier,
    entry.mode,
    entry.overlay,
    entry.rung,
  ]
    .map((part) => part ?? "-")
    .join(":");

console.log("Memory-contract reconciliation report");
console.log("Nothing here fails a build. This is a worklist, not a gate (Michael, 2026-08-17).");
if (result.unavailable) {
  console.log(`\n  Reconciliation did not run: ${result.unavailable}`);
  console.log("\nDone. Exit 0 always.");
  process.exit(0);
}
if (result.buildIncomplete) {
  console.log(
    "\n  NOTE: the reconciliation below is complete, but the matrix build stopped afterwards on an\n" +
      `  unrelated invariant: ${result.buildIncomplete}`,
  );
}
console.log(
  `\n  ${plural(result.providers ?? 0, "engine-declared provider contract")}, ` +
    `${plural(result.bespokeWaivers ?? 0, "engine-declared bespoke route")}.`,
);
console.log(
  `  ${plural(findings.length, "mismatch", "mismatches")}${legFilter ? ` on leg ${legFilter}` : ""}.`,
);

for (const [leg, count] of Object.entries(result.byLeg ?? {})) {
  console.log(`    ${leg}: ${count}`);
}

const byLegThenDirection = new Map();
for (const entry of findings) {
  const key = `${entry.leg} / ${entry.direction}`;
  if (!byLegThenDirection.has(key)) byLegThenDirection.set(key, []);
  byLegThenDirection.get(key).push(entry);
}

for (const [key, entries] of [...byLegThenDirection].sort(([a], [b]) => a.localeCompare(b))) {
  console.log(`\n${key} — ${plural(entries.length, "coordinate")}`);
  console.log("  backend:provider:modelId:tier:mode:overlay:rung");
  for (const entry of entries.map(coordinate).sort()) {
    console.log(`    ${entry}`);
  }
}

console.log("\nDone. Exit 0 always.");
