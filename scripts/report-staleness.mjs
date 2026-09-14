#!/usr/bin/env node
// What is out of date? — decode-quality measurements, licensing, calibration evidence.
//
// WHY THIS EXISTS
// ---------------
// Michael, 2026-08-15: "we need some way to know what needs fingerprints and measurements and
// licenses, so there should be some kind of scripts that we can run to determine what's out of
// date, but no, I don't want any tests/ci or especially runtimes failing because of those missing."
//
// So this REPORTS and always exits 0. It is deliberately not wired into CI, and nothing here should
// ever grow an exit code. The point of sc-19728 and sc-19751 was that the gates guarding this
// information fired on unrelated changes — an inference pin bump invalidated every decode-quality
// measurement and simultaneously demanded a licensing re-audit — so the answer is to make the
// information *legible on demand* rather than enforced continuously. A human runs this, reads it,
// and decides what to remeasure. Recording is not enforcing.
//
// Usage:
//   node scripts/report-staleness.mjs
//   node scripts/report-staleness.mjs --fingerprint <sha256>
//
// `--fingerprint` is the current inference decode-quality source closure, printed by
// `scripts/ci/decode_quality_implementation_fingerprint.py` in a checked-out inference tree at the
// revision this repo pins. Without it the stamps are still listed and grouped; they just are not
// labelled current or historical, and the report says so rather than guessing.

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_CATALOG = path.join(ROOT, "config/manifests/builtin.models.jsonc");
const EVIDENCE = path.join(ROOT, "docs/generated/memory-calibration-evidence.json");

const FINGERPRINT_FLAG = "--fingerprint";
const CATALOG_FLAG = "--catalog";
const argv = process.argv.slice(2);
const fingerprintAt = argv.indexOf(FINGERPRINT_FLAG);
const currentFingerprint = fingerprintAt >= 0 ? argv[fingerprintAt + 1] : null;
const catalogAt = argv.indexOf(CATALOG_FLAG);
// The decode-quality rows live on whichever branch packaged them, so the report has to be able to
// read a catalog other than this checkout's — that is also what makes it testable against fixtures.
const CATALOG = catalogAt >= 0 ? path.resolve(argv[catalogAt + 1] ?? "") : DEFAULT_CATALOG;

if (fingerprintAt >= 0 && !/^[0-9a-f]{64}$/.test(currentFingerprint ?? "")) {
  console.error(`${FINGERPRINT_FLAG} expects a lowercase 64-hex digest`);
  process.exit(2); // a malformed invocation, not a finding
}

const section = (title) => console.log(`\n${title}\n${"=".repeat(title.length)}`);
const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

function loadCatalog() {
  const parsed = JSON.parse(stripJsoncComments(readFileSync(CATALOG, "utf8")));
  return Array.isArray(parsed) ? parsed : (parsed.models ?? []);
}

/** Every declared decode-quality row, flattened with the model it belongs to. */
function decodeQualityRows(models) {
  const rows = [];
  const unmeasured = new Map();
  for (const model of models) {
    for (const backend of ["mlx", "candle"]) {
      for (const impl of model[backend]?.memoryStrategyContract?.implementations ?? []) {
        const declared = impl.parameterRanges?.decodeGeometryPolicies ?? [];
        for (const row of declared) rows.push({ model: model.id, backend, row });
        const tilesDecode =
          impl.strategy === "BoundedDecode" ||
          impl.parameterRanges?.decodeTileEdges?.length > 0;
        if (tilesDecode && declared.length === 0) {
          // One entry per model+backend, not per numeric tier — three identical lines for the same
          // model is noise in a report a human is meant to read.
          const key = `${model.id} (${backend})`;
          unmeasured.set(key, (unmeasured.get(key) ?? 0) + 1);
        }
      }
    }
  }
  return { rows, unmeasured };
}

function reportDecodeQuality(models) {
  section("Decode-quality measurements");
  const { rows, unmeasured } = decodeQualityRows(models);

  if (rows.length === 0) {
    console.log("  No decode-quality rows are packaged in this branch's catalog.");
  } else {
    const byStamp = new Map();
    for (const entry of rows) {
      const stamp = entry.row.implementationFingerprint ?? "(none)";
      if (!byStamp.has(stamp)) byStamp.set(stamp, []);
      byStamp.get(stamp).push(entry);
    }
    console.log(`  ${plural(rows.length, "row")} across ${plural(byStamp.size, "distinct source stamp")}.`);
    for (const [stamp, entries] of [...byStamp].sort((a, b) => b[1].length - a[1].length)) {
      const models = new Set(entries.map((entry) => entry.model));
      const label = currentFingerprint
        ? stamp === currentFingerprint
          ? "CURRENT"
          : "measured by different source"
        : "not compared";
      console.log(
        `    ${stamp.slice(0, 12)}…  ${String(entries.length).padStart(3)} rows  ` +
          `${String(models.size).padStart(2)} models  [${label}]`,
      );
      if (currentFingerprint && stamp !== currentFingerprint) {
        console.log(`        ${[...models].sort().join(", ")}`);
      }
    }
    if (!currentFingerprint) {
      console.log(
        `\n  Pass ${FINGERPRINT_FLAG} <sha256> to label these current or historical. Derive it by running\n` +
          "  scripts/ci/decode_quality_implementation_fingerprint.py in an inference checkout at the pinned revision.",
      );
    }

    // The measured verdict matters more than the stamp, and is easy to lose behind it: a row can be
    // perfectly current and still say tiled decode was worse than the tolerance at that coordinate.
    const verdicts = new Map();
    for (const entry of rows) {
      const bucket = verdicts.get(entry.model) ?? { admitted: 0, refused: 0 };
      bucket[entry.row.disposition?.kind === "refused" ? "refused" : "admitted"] += 1;
      verdicts.set(entry.model, bucket);
    }
    const admitted = [...verdicts.values()].reduce((sum, bucket) => sum + bucket.admitted, 0);
    console.log(`\n  Measured verdicts: ${admitted} admitted, ${rows.length - admitted} refused on quality.`);
    for (const [model, bucket] of [...verdicts].sort()) {
      const flag = bucket.admitted === 0 ? "  <- no coordinate admitted" : "";
      console.log(
        `    ${model.padEnd(22)} admitted ${String(bucket.admitted).padStart(2)}  ` +
          `refused ${String(bucket.refused).padStart(2)}${flag}`,
      );
    }
    if (admitted < rows.length) {
      console.log(
        "\n  A refused row is a MEASUREMENT, not a staleness problem: tiled decode differed from dense\n" +
          "  decode by more than the recorded tolerance at that coordinate, so the route falls back to\n" +
          "  dense. Remeasuring will not change it; improving the tiling or widening the tolerance will.",
      );
      const worst = rows
        .filter((entry) => entry.row.disposition?.kind === "refused")
        .slice(0, 5);
      console.log("\n  First few refusal reasons:");
      for (const entry of worst) console.log(`    ${entry.model}: ${entry.row.disposition.reason}`);
      if (rows.length - admitted > worst.length) {
        console.log(`    … and ${rows.length - admitted - worst.length} more.`);
      }
    }
  }

  if (unmeasured.size > 0) {
    console.log(`\n  ${plural(unmeasured.size, "route")} tile the VAE decode but carry no measurement:`);
    for (const [label, implementations] of [...unmeasured].sort()) {
      console.log(`    ${label}  (${plural(implementations, "implementation")})`);
    }
  }

  console.log(
    "\n  A differing stamp is NOT a defect and nothing refuses it (sc-19728). It records which source\n" +
      "  closure produced the measurement, so you can decide whether a change since then could have\n" +
      "  altered decode output. Measurements stand until they are remeasured.",
  );
}

function reportLicensing() {
  section("Licensing");
  // Delegate rather than re-implement: check-license-coverage.mjs owns coverage, stale ids,
  // duplicates, document wiring, the pinned source audit, and crate classification. Since sc-19751
  // it reports instead of gating, so its findings are exactly this section's content.
  const result = spawnSync("node", [path.join(ROOT, "scripts/check-license-coverage.mjs")], {
    encoding: "utf8",
  });
  const output = `${result.stdout ?? ""}${result.stderr ?? ""}`.trimEnd();
  console.log(
    output
      .split("\n")
      .map((line) => `  ${line}`)
      .join("\n"),
  );
  if (result.status !== 0) {
    console.log(
      "\n  NOTE: check-license-coverage exited non-zero, which it should not do since sc-19751.",
    );
  }
}

function reportCalibrationEvidence() {
  section("Memory-calibration evidence");
  let bundle;
  try {
    bundle = JSON.parse(readFileSync(EVIDENCE, "utf8"));
  } catch {
    console.log("  No generated evidence bundle on this branch (docs/generated/memory-calibration-evidence.json).");
    return;
  }
  const records = bundle.records ?? [];
  console.log(`  ${plural(records.length, "record")} in the corpus.`);
  console.log(
    "  Per-record staleness is graded at runtime against each provider's calibration fingerprint and\n" +
      "  ABI, not here. sc-22512 retired the corpus-shrink gate that used to be named on this line:\n" +
      "  a record leaving the corpus is an unmeasured cell, and an unmeasured cell falls back to the\n" +
      "  conservative analytic estimate rather than failing anything.",
  );
}

const models = loadCatalog();
console.log(`Staleness report — ${plural(models.length, "catalog model")}`);
console.log("Nothing here fails a build. This is a worklist, not a gate (sc-19751).");
reportDecodeQuality(models);
reportLicensing();
reportCalibrationEvidence();
console.log("\nDone. Exit 0 always.");
