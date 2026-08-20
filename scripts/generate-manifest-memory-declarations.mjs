#!/usr/bin/env node
// Regenerate the manifest's engine-derived memory declarations (sc-20246, epic 18304).
//
// Run via `npm run generate:manifest-memory-declarations`. Rewrites
// `config/manifests/builtin.models.jsonc` in place, touching ONLY the marked generated regions
// inside per-model memory-declaration blocks. Every hand-authored row, limit, download, UI field and
// comment in the file is preserved byte-for-byte.
//
// Inputs are all CHECKED IN, so this runs on any machine with no engine linked and no weights:
//   1. config/engine-capabilities/capabilities.{mlx,candle}.json — the stage-1 dumps of the linked
//      provider registries, including the `memoryContracts` / `memoryRouteWitnesses` inventory
//      (PR #2386). Produced by `cargo run -p sceneworks-worker --bin dump-engine-capabilities` on a
//      lane that links engines; NOT re-dumped here.
//   2. crates/sceneworks-worker/src/engines.rs — MODEL_TABLE, the SceneWorks-id -> engine-id join.
//   3. crates/sceneworks-worker/src/image_jobs/strict_control.rs — STRICT_CONTROL_ENGINES, which is
//      how a `*_control` provider is known to be a real overlay on a base model's route rather than
//      a separately routed entry.
//   4. crates/sceneworks-worker/src/image_jobs/base.rs — WIRED_{MLX,CANDLE}_POSE_FAMILIES, the wired
//      control lanes, for the catalog-reach intersection (see `catalogAxes`).
//   5. crates/sceneworks-worker/src/memory_route_registry.rs — RULES, for both the deferred-route
//      population's `legacy_shaping` flag (see `parseRequestContextLanes`: an MLX lane that requires
//      `requestContexts` cannot take a projected row at all) and the vocabularies the witnesses use.
//
// Why the engine and not the manifest is authoritative, what is and is not projected, and why
// generated rows are appended rather than merged, are all documented on
// `scripts/lib/manifest-memory-declarations.mjs`.
//
// NOT A GATE, and there is no check mode. Nothing verifies this output and nothing may:
// `npm run report:memory-contract-reconciliation` is BOTH the disagreement worklist AND the freshness
// signal — it prints whether the committed manifest is still the projection of the committed dumps,
// names this script as the refresh, and always exits 0 (Michael, 2026-08-17). A blocking fixed-point
// invariant was deliberately NOT added; a stale projection is a thing a human reads about in the
// report, never a red build. Do not wire this into CI, and do not add `--check`.
//
// Usage:
//   node scripts/generate-manifest-memory-declarations.mjs
//   node scripts/generate-manifest-memory-declarations.mjs --dry-run   # report only, write nothing
//   node scripts/generate-manifest-memory-declarations.mjs --clear     # remove generated regions

import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { clearProjection, projectManifestBody } from "./lib/manifest-memory-declarations.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const dryRun = argv.includes("--dry-run");
const clearOnly = argv.includes("--clear");

const MANIFEST = "config/manifests/builtin.models.jsonc";
const ENGINES = "crates/sceneworks-worker/src/engines.rs";
const STRICT_CONTROL = "crates/sceneworks-worker/src/image_jobs/strict_control.rs";
const IMAGE_ROUTING = "crates/sceneworks-worker/src/image_jobs/base.rs";
const ROUTE_REGISTRY = "crates/sceneworks-worker/src/memory_route_registry.rs";
const BACKENDS = ["mlx", "candle"];

const read = (relative) => readFileSync(path.join(ROOT, relative), "utf8");

const body = read(MANIFEST);

if (clearOnly) {
  const cleared = clearProjection(body);
  if (!dryRun) writeFileSync(path.join(ROOT, MANIFEST), cleared, "utf8");
  console.log(
    `${MANIFEST}: generated regions ${dryRun ? "would be" : ""} removed ` +
      `(${body.length - cleared.length} bytes)`,
  );
  process.exit(0);
}

const engineFacts = BACKENDS.map((backend) =>
  JSON.parse(read(`config/engine-capabilities/capabilities.${backend}.json`)),
);
const pins = [...new Set(engineFacts.map((facts) => facts.generatedFrom?.inferenceRevision))];
if (pins.length !== 1 || !pins[0]) {
  throw new Error(
    `the ${BACKENDS.join("/")} capability dumps are keyed to different inference revisions ` +
      `(${pins.join(", ")}); re-dump the stale lane before projecting`,
  );
}

const result = projectManifestBody({
  body,
  engineFacts,
  enginesSource: read(ENGINES),
  strictControlSource: read(STRICT_CONTROL),
  imageRoutingSource: read(IMAGE_ROUTING),
  routeRegistrySource: read(ROUTE_REGISTRY),
});

if (!dryRun) writeFileSync(path.join(ROOT, MANIFEST), result.body, "utf8");

const rows = result.plans.reduce((count, plan) => count + plan.rows.length, 0);
console.log(`Manifest memory declarations projected from the ${pins[0]} capability dumps`);
console.log(
  `  ${rows} generated implementation rows across ${result.plans.length} model/backend blocks` +
    `${dryRun ? " (dry run — nothing written)" : ` -> ${MANIFEST}`}`,
);

const newContracts = result.plans.filter((plan) => !plan.hasContract);
if (newContracts.length) {
  console.log(`\n  ${newContracts.length} backend blocks gained a memoryStrategyContract:`);
  for (const plan of newContracts) {
    console.log(
      `    ${plan.modelId}:${plan.backend} provider=${plan.contractProvider} ` +
        `(${plan.rows.length} rows)`,
    );
  }
}

if (result.withholds.length) {
  console.log("\n  Deliberate withholds honored (declared LESS than the engine dumps, on purpose):");
  for (const entry of result.withholds) {
    const rungs = entry.rungs === "all" ? "all rungs" : entry.rungs.join(", ");
    console.log(
      `    ${entry.backend}:${entry.provider} on ${entry.modelId} — ${rungs} ` +
        // Both fields are required by `withheldRungs` and by the authoring schema, so there is no
        // "no story" fallback to print: an uncited withhold never reaches here.
        `(${entry.declaration.story}: ${entry.declaration.reason})`,
    );
  }
} else {
  console.log("\n  Deliberate withholds honored: none declared.");
}

if (result.unhosted.length) {
  console.log(
    "\n  Engine providers with no image-manifest host — these CANNOT be declared through this file " +
      "and stay on the reconciliation report:",
  );
  for (const entry of result.unhosted) {
    console.log(`    ${entry.backend}:${entry.provider} (${entry.rungs.join(", ")})`);
  }
}

// The honest residue: rungs the engine implements at coordinates production cannot reach. Declaring
// these would claim an unreachable capability, so they stay on the reconciliation report as genuine
// route-vs-engine work.
const REASONS = {
  "requires-request-contexts":
    "the MLX lane requires `requestContexts` on every declaration row, which no dump publishes",
  "tier-not-advertised": "the catalog entry does not advertise the tier on this backend",
  "no-route-witness": "the deferred-route registry witnesses no route at that tier",
  "no-catalog-coordinate": "no witnessed mode/overlay survives the catalog's own axes",
};
for (const reason of Object.keys(REASONS)) {
  const gaps = result.skipped.filter((entry) => entry.reason === reason);
  if (!gaps.length) continue;
  const byLane = new Map();
  for (const entry of gaps) {
    const key = `${entry.backend}:${entry.provider}`;
    if (!byLane.has(key)) byLane.set(key, []);
    byLane.get(key).push(`${entry.rung}[${entry.tiers.join(",")}]`);
  }
  console.log(`\n  ${gaps.length} rung/tier groups NOT declared — ${REASONS[reason]}:`);
  for (const key of [...byLane.keys()].sort()) {
    console.log(`    ${key} ${byLane.get(key).join(" ")}`);
  }
}

console.log(
  "\nNext: `npm run report:memory-contract-reconciliation` for what still disagrees, then " +
    "regenerate the derived artifacts (`npm run generate:memory-matrix`, " +
    "`npm --prefix apps/web run gen:preview-support`).",
);
