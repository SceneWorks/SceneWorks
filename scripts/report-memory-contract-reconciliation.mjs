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
//   node scripts/report-memory-contract-reconciliation.mjs --drift   # only what two sources disagree on
//
// It also carries the FRESHNESS signal for the manifest's engine-projected memory declarations
// (sc-20246): whether `config/manifests/builtin.models.jsonc` is still the projection of the committed
// capability dumps. That lives here, not in a test, because a blocking fixed-point invariant would be
// a new gate. See `manifestFixedPoint` below.

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { planProjection, projectManifestBody } from "./lib/manifest-memory-declarations.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import { reconciliationMismatchKey } from "./lib/memory-contract-reconciliation.mjs";
import { triageMemoryContractMismatches } from "./lib/memory-contract-triage.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const asJson = argv.includes("--json");
const legAt = argv.indexOf("--leg");
const legFilter = legAt >= 0 ? argv[legAt + 1] : null;
const driftOnly = argv.includes("--drift");

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
    const engineFacts = ["mlx", "candle"].map((backend) => {
      const facts = JSON.parse(read(`config/engine-capabilities/capabilities.${backend}.json`));
      // Refuse to answer off a dump that does not carry the two surfaces the projection reads. The
      // projector treats a missing inventory as "nothing to project", which would render an unusable
      // dump as a STALE manifest — blaming the wrong artifact, and telling the reader to regenerate
      // something that is already correct.
      for (const surface of ["memoryContracts", "memoryRouteWitnesses"]) {
        if (!Array.isArray(facts[surface]) || facts[surface].length === 0) {
          throw new Error(`the ${backend} capability dump has no ${surface} inventory`);
        }
      }
      return facts;
    });
    const projected = projectManifestBody({
      body,
      engineFacts,
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

/**
 * The projection plan, for the triage join (sc-21505).
 *
 * The engine_manifest leg reports that the manifest carries no declaration; the projector is what
 * knows WHY it could not write one. Degrades to an empty plan on any input problem, exactly like
 * `manifestFixedPoint` — an unavailable explanation must not turn a report into a failure, and an
 * empty plan classifies those findings `unclassified`, which is visible rather than silent.
 */
function projectionPlan() {
  const read = (relative) => readFileSync(path.join(ROOT, relative), "utf8");
  try {
    return planProjection({
      manifest: JSON.parse(stripJsoncComments(read("config/manifests/builtin.models.jsonc"))),
      engineFacts: ["mlx", "candle"].map((backend) =>
        JSON.parse(read(`config/engine-capabilities/capabilities.${backend}.json`)),
      ),
      enginesSource: read("crates/sceneworks-worker/src/engines.rs"),
      strictControlSource: read("crates/sceneworks-worker/src/image_jobs/strict_control.rs"),
      imageRoutingSource: read("crates/sceneworks-worker/src/image_jobs/base.rs"),
      routeRegistrySource: read("crates/sceneworks-worker/src/memory_route_registry.rs"),
    });
  } catch (error) {
    // Same degrade-never-throw contract as `manifestFixedPoint`, but the reason travels with it: an
    // empty plan reclassifies every engine_manifest finding as `unclassified`, and a reader must not
    // have to guess whether that means "genuinely unexplained" or "the projection would not load".
    return { unhosted: [], skipped: [], error: (error?.message ?? String(error)).split("\n")[0] };
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

// `process.exit()` is deliberately ABSENT from this file (sc-20246).
//
// `console.log` on a PIPE is asynchronous, so `process.exit(0)` immediately after writing tears the
// process down before stdout drains: `--json` piped to another process lost everything past 128 KiB
// (131072 of 190801 bytes, truncated mid-string), while the same output redirected to a file was
// whole — a difference that quietly corrupts any consumer that pipes. Every path below RETURNS from
// `main` instead, so node exits naturally once the stream has flushed. `main` never throws, so the
// implicit exit code is 0, which is the always-exit-0 contract this report is built on.
function main() {
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
    return;
  }

  // Triage BEFORE the leg/drift filters, so every finding is classified against the whole
  // enumeration and a filter narrows what is printed rather than what was reasoned about.
  const plan = projectionPlan();
  const triage = triageMemoryContractMismatches(result.findings ?? [], plan);
  const classOf = new Map();
  for (const group of triage.classes) {
    for (const entry of group.findings) classOf.set(reconciliationMismatchKey(entry), group);
  }
  const findings = (result.findings ?? [])
    .map((entry) => ({ ...entry, triageClass: classOf.get(reconciliationMismatchKey(entry))?.name ?? null }))
    .filter((entry) => !legFilter || entry.leg === legFilter)
    .filter(
      (entry) =>
        !driftOnly || classOf.get(reconciliationMismatchKey(entry))?.disposition === "drift",
    );

  if (asJson) {
    console.log(
      JSON.stringify(
        {
          ...result,
          manifestProjection: manifestFixedPoint(),
          triage: {
            total: triage.total,
            byDisposition: triage.byDisposition,
            // Findings are already on `findings`; the summary carries only the counts and the rule.
            classes: triage.classes.map(({ findings: _findings, ...group }) => group),
          },
          findings,
        },
        null,
        2,
      ),
    );
    return;
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
    return;
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
    `  ${plural(findings.length, "mismatch", "mismatches")}${legFilter ? ` on leg ${legFilter}` : ""}` +
      `${driftOnly ? " that are genuine drift" : ""}.`,
  );

  // Derived from the FILTERED findings rather than `result.byLeg`, so the breakdown and the headline
  // count above always describe the same set. Reading the unfiltered summary here made `--leg` print a
  // total that contradicted its own headline.
  const legTotals = new Map();
  for (const entry of findings) legTotals.set(entry.leg, (legTotals.get(entry.leg) ?? 0) + 1);
  for (const leg of [...legTotals.keys()].sort()) {
    console.log(`    ${leg}: ${legTotals.get(leg)}`);
  }

  // The triage split (sc-21505). Printed BEFORE the per-leg enumeration, because the first question
  // a reader has about a four-hundred-item list is how much of it is work.
  if (plan.error) {
    console.log(
      `\n  NOTE: the declaration projection could not be read (${plan.error}), so every\n` +
        "  engine_manifest finding below is classified `unclassified` for want of a reason, not\n" +
        "  because it is unexplained.",
    );
  }
  console.log(
    `\n  Triage: ${triage.byDisposition.drift} genuine drift, ` +
      `${triage.byDisposition["by-construction"]} by construction.`,
  );
  for (const group of triage.classes) {
    console.log(`    [${group.disposition}] ${group.count} — ${group.title}`);
  }
  console.log(
    "\n  by-construction: the coordinate cannot carry a declaration, or the two sides are keyed at\n" +
      "  different grains. Writing declarations here would claim unreachable capability. Count of work: 0.\n" +
      "  drift: the two sides make contradictory claims about the same fact, so one of them is wrong.\n" +
      "  Run with --drift for just those. Rationale per class: scripts/lib/memory-contract-triage.mjs.",
  );

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
}

main();
