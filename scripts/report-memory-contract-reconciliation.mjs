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
//
// It also carries the FRESHNESS signal for the manifest's engine-projected memory declarations
// (sc-20246): whether `config/manifests/builtin.models.jsonc` is still the projection of the committed
// capability dumps. That lives here, not in a test, because a blocking fixed-point invariant would be
// a new gate. See `manifestFixedPoint` below.

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { projectManifestBody } from "./lib/manifest-memory-declarations.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const asJson = argv.includes("--json");
const legAt = argv.indexOf("--leg");
const legFilter = legAt >= 0 ? argv[legAt + 1] : null;

/**
 * Is the committed manifest still the projection of the committed dumps? (sc-20246)
 *
 * THIS IS THE FRESHNESS SIGNAL, and it is deliberately here rather than in a test. A blocking
 * "the committed projection is a fixed point" invariant would be a new gate — ruled out by Michael's
 * 2026-08-17 decision, and it would contradict the projector's own not-a-gate contract. So the report
 * says whether the manifest is stale and what to run; a human decides. Never throws: a broken input
 * degrades to "could not evaluate", because an unavailable freshness signal is not a build failure.
 */
function manifestFixedPoint() {
  const read = (relative) => readFileSync(path.join(ROOT, relative), "utf8");
  try {
    const body = read("config/manifests/builtin.models.jsonc");
    const projected = projectManifestBody({
      body,
      engineFacts: ["mlx", "candle"].map((backend) =>
        JSON.parse(read(`config/engine-capabilities/capabilities.${backend}.json`)),
      ),
      enginesSource: read("crates/sceneworks-worker/src/engines.rs"),
      strictControlSource: read("crates/sceneworks-worker/src/image_jobs/strict_control.rs"),
      imageRoutingSource: read("crates/sceneworks-worker/src/image_jobs/base.rs"),
      routeRegistrySource: read("crates/sceneworks-worker/src/memory_route_registry.rs"),
    });
    return { current: projected.body === body };
  } catch (error) {
    return { error: (error?.message ?? String(error)).split("\n")[0] };
  }
}

function fixedPointLine() {
  const state = manifestFixedPoint();
  if (state.error) {
    return `  Manifest projection freshness: could not evaluate — ${state.error}`;
  }
  return state.current
    ? "  Manifest projection: the memory declarations ARE the projection of the committed engine dumps."
    : "  Manifest projection: STALE — the memory declarations are not the projection of the committed\n" +
      "  engine dumps. Run `npm run generate:manifest-memory-declarations` to refresh, then regenerate\n" +
      "  the derived artifacts. Findings below may reflect the old projection.";
}

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
  console.log(fixedPointLine());
  console.log("  Could not assemble the reconciliation inputs on this branch:");
  console.log(`  ${(error?.stderr || error?.message || String(error)).toString().trim().split("\n").slice(-4).join("\n  ")}`);
  console.log("\nDone. Exit 0 always.");
  process.exit(0);
}

const findings = (result.findings ?? []).filter(
  (entry) => !legFilter || entry.leg === legFilter,
);

if (asJson) {
  console.log(
    JSON.stringify({ ...result, manifestProjection: manifestFixedPoint(), findings }, null, 2),
  );
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
console.log(`\n${fixedPointLine()}`);
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

// Derived from the FILTERED findings rather than `result.byLeg`, so the breakdown and the headline
// count above always describe the same set. Reading the unfiltered summary here made `--leg` print a
// total that contradicted its own headline.
const legTotals = new Map();
for (const entry of findings) legTotals.set(entry.leg, (legTotals.get(entry.leg) ?? 0) + 1);
for (const leg of [...legTotals.keys()].sort()) {
  console.log(`    ${leg}: ${legTotals.get(leg)}`);
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
