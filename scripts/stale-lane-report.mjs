#!/usr/bin/env node

/**
 * sc-18098 — the stale-lane batch report (epic 18093 R3).
 *
 * WHAT THIS IS FOR
 *
 * Measurement currency used to be treated as a correctness gate: a lane whose provider compile
 * closure had moved was DEMOTED, and the only remedy was a re-capture. Epic 18093 retired that.
 * `crates/sceneworks-worker/src/ladder_margin_policy.rs` (sc-18095/18096/18097) keeps stale-closure
 * measured evidence ELIGIBLE, serving its measured numbers behind a widened admission margin, and
 * admits estimate-backed candidates behind a wider one still. Currency is therefore a SIGNAL about
 * how much conservatism the runtime is currently buying, not a per-PR obligation.
 *
 * A signal needs somewhere to be read. That is this script: an on-demand batch view of which lanes
 * are stale, how much of the corpus and of the shipped admission surface each one covers, and the
 * margin widening the runtime is applying to it right now. Nothing here runs in `npm run check`,
 * `npm run rust:check`, the pre-push hook or CI — wiring it into a gate would rebuild exactly the
 * per-PR pressure R3 exists to remove.
 *
 * WHAT "STALE" MEANS HERE — the same predicate the runtime uses, not a lookalike
 *
 * A lane is `<backend>:<provider>`, keyed exactly as `config/inference-provider-closures.json`,
 * `memory-calibration-harness.mjs#evidenceSemantics` and `generate-memory-matrix.mjs#closureIsCurrent`
 * key it. A record or manifest calibration binding is stale when the closure digest it was captured
 * under differs from the live digest for its own lane. The live table is loaded through
 * `validatedInferenceClosures`, the SAME gate the matrix generator uses, so the report cannot report
 * currency against a closure table that is not keyed to the live Cargo pin.
 *
 * TWO POPULATIONS, KEPT SEPARATE
 *
 *   RECORDS  — `docs/generated/memory-calibration-evidence.json`. The measurement corpus. What a
 *              re-capture would have to reproduce.
 *   BINDINGS — every `inferenceClosureDigest` in `config/manifests/builtin.models.jsonc`. The
 *              SHIPPED admission surface: these are what the worker's fit gates actually consult, so
 *              a stale binding is a production decision running under a widened margin today, while
 *              a stale record is only corpus debt. They are ranked in that order for that reason.
 *
 * THE BINDING POPULATION IS THE LOCATOR'S, NOT THIS FILE'S
 *
 * The first version of this report walked `<model>.<backend>.calibrations[]` and called it the
 * admission surface. That is 31 of the manifest's 33 bindings. The missing two are `turboFit` and
 * `candle.control` — whole-block fits that sit OUTSIDE `calibrations[]`, are read by
 * `vram_gate.rs` and `krea_control_fit.rs`, and are exactly the pair sc-17989 dragged back under the
 * closure gate after they had sat outside it for the whole of sc-17774. Re-deriving the population
 * reopened that hole in a new file: `candle:krea_2_turbo` and `candle:krea_2_turbo_control` printed
 * as "declared but never captured" while both were stale production bindings.
 *
 * So the population comes from `backfill-closure-digests.mjs#stampManifest` — the CI gate's own
 * locator, brace-depth accurate and orphan-checked — and this file refuses to run if that locator
 * reports a skipped or orphaned digest. Model attribution (the MODELS column) is read off the
 * parsed manifest and cross-checked against the locator's per-lane counts, so the cosmetic walk
 * cannot quietly disagree with the authoritative one.
 *
 * MARGIN: DERIVED, NEVER RESTATED
 *
 * The widening column is `staleMeasuredMargin` from `scripts/derive-ladder-margins.mjs`, computed
 * over the same evidence corpus this report reads. That module's constants are pinned against
 * `crates/sceneworks-worker/src/ladder_margin_policy.rs` by `scripts/derive-ladder-margins.test.mjs`,
 * so the number printed here is the number the runtime applies, and a drift on either side reds that
 * test. No margin literal appears in this file.
 *
 * Run: `node scripts/stale-lane-report.mjs [--json]`  (`npm run report:stale-lanes`)
 */

import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { stampManifest } from "./backfill-closure-digests.mjs";
import { deriveMargins } from "./derive-ladder-margins.mjs";
import { validatedInferenceClosures } from "./generate-memory-matrix.mjs";
import { inferencePinFromCargo } from "./inference-closure-digest.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export const SOURCE_PATHS = Object.freeze({
  closures: "config/inference-provider-closures.json",
  evidence: "docs/generated/memory-calibration-evidence.json",
  manifest: "config/manifests/builtin.models.jsonc",
  cargo: "Cargo.toml",
});

/** Provenance for the margin column, printed so a reader can check it rather than trust it. */
export const MARGIN_SOURCE =
  "scripts/derive-ladder-margins.mjs#staleMeasuredMargin (pinned against " +
  "crates/sceneworks-worker/src/ladder_margin_policy.rs by scripts/derive-ladder-margins.test.mjs)";

export function laneOf(backend, provider) {
  return `${backend}:${provider}`;
}

/**
 * Which catalog models declare a binding on each lane, for the MODELS column only.
 *
 * A binding is ANY object under a model's `mlx`/`candle` block carrying both `provider` and
 * `inferenceRevision` — the same shape `stampManifest`'s locator pairs on, at any depth, so
 * `turboFit` and `candle.control` are included by construction rather than by enumeration. This is
 * attribution, not population: `manifestBindings` cross-checks the per-lane counts against the
 * locator and throws if the two walks disagree.
 */
export function laneModelAttribution(manifest) {
  const byLane = new Map();
  const visit = (value, lane) => {
    if (Array.isArray(value)) {
      for (const item of value) visit(item, lane);
      return;
    }
    if (value === null || typeof value !== "object") return;
    if (typeof value.provider === "string" && typeof value.inferenceRevision === "string") {
      lane(value.provider);
    }
    for (const child of Object.values(value)) visit(child, lane);
  };
  for (const model of manifest.models ?? []) {
    for (const backend of ["mlx", "candle"]) {
      if (!model[backend]) continue;
      visit(model[backend], (provider) => {
        const key = laneOf(backend, provider);
        if (!byLane.has(key)) byLane.set(key, { models: new Set(), count: 0 });
        byLane.get(key).models.add(model.id);
        byLane.get(key).count += 1;
      });
    }
  }
  return byLane;
}

/**
 * Every manifest closure binding, flattened to `{ lane, modelId, digest }`.
 *
 * The population is `stampManifest`'s — the CI gate's own locator (see the module header for why
 * this file must not re-derive it). Its coverage checks are load-bearing here too: a `skipped`
 * revision or an `orphan` digest means part of the admission surface is invisible, and a report that
 * silently omitted it would understate exactly what it exists to surface.
 */
export function manifestBindings({ manifestBody, manifest }) {
  // The locator is a stamper; it is used here purely as a scanner, so the rewritten body is
  // discarded and the replacement digest is an inert constant that never reaches disk.
  const { located, skipped, orphans } = stampManifest(manifestBody, () => "0".repeat(64));
  if (skipped.length || orphans.length) {
    throw new Error(
      "the manifest closure locator cannot see the whole admission surface, so the stale-lane " +
        `report would understate it:\n  ${[...skipped, ...orphans].join("\n  ")}`,
    );
  }
  const attribution = laneModelAttribution(manifest);
  const counted = new Map();
  for (const binding of located) counted.set(binding.key, (counted.get(binding.key) ?? 0) + 1);
  for (const lane of new Set([...counted.keys(), ...attribution.keys()])) {
    const authoritative = counted.get(lane) ?? 0;
    const attributed = attribution.get(lane)?.count ?? 0;
    if (authoritative !== attributed) {
      throw new Error(
        `manifest binding walks disagree on ${lane}: the locator found ${authoritative}, the parsed ` +
          `attribution found ${attributed}. One of the two is missing a binding shape.`,
      );
    }
  }
  return located.map((binding) => ({
    lane: binding.key,
    modelId: [...(attribution.get(binding.key)?.models ?? [])].sort().join("+") || null,
    digest: binding.digest,
  }));
}

/** Every evidence record, flattened to `{ lane, modelId, digest }`. */
export function evidenceBindings(records) {
  return records.map((record) => ({
    lane: laneOf(record.backend, record.target?.provider),
    modelId: record.target?.modelId ?? null,
    digest: record.repositories?.inference?.closureDigest ?? null,
  }));
}

function tally(items, liveDigest) {
  const stale = items.filter((item) => item.digest !== liveDigest);
  return {
    total: items.length,
    stale: stale.length,
    current: items.length - stale.length,
    staleItems: stale,
  };
}

function shortDigests(items) {
  return [...new Set(items.map((item) => item.digest ?? "(absent)"))]
    .sort()
    .map((digest) => (digest === "(absent)" ? digest : digest.slice(0, 12)));
}

/**
 * Rank stale lanes by impact.
 *
 * The ordering is lexicographic over quantities that are all reported, so a consumer that disagrees
 * with the weighting can re-rank from `--json` rather than being stuck with it:
 *
 *   1. `widenedAdmissionSurface` = stale BINDINGS x the margin the runtime widens them by. Bindings
 *      are the shipped admission surface, so this is production impact and it leads.
 *   2. `widenedEvidenceSurface`  = stale RECORDS x the same margin. Corpus debt: what a re-capture
 *      of this lane would actually have to cover.
 *   3. lane name, so the ordering is total and the output is diffable.
 *
 * Multiplying a count by the margin rather than ranking on the count alone is the "margin-widening
 * impact" the epic asks for: two lanes with equal stale counts are not equally costly when one runs
 * on the MLX margin and the other on candle's, which is 2.5x narrower.
 */
export function rankLanes(lanes) {
  return [...lanes].sort(
    (left, right) =>
      right.impact.widenedAdmissionSurface - left.impact.widenedAdmissionSurface ||
      right.impact.widenedEvidenceSurface - left.impact.widenedEvidenceSurface ||
      left.lane.localeCompare(right.lane),
  );
}

/**
 * Build the report.
 *
 * @param liveDigests  `Map<lane, digest>` from `validatedInferenceClosures`.
 * @param declarations the `providers` block of the closure config (for the crate pointer).
 * @param records      the evidence bundle's records.
 * @param manifest     the parsed builtin model manifest (model attribution only).
 * @param manifestBody the raw JSONC body — the authoritative binding population is located in it.
 */
export function buildStaleLaneReport({ liveDigests, declarations, records, manifest, manifestBody, meta = {} }) {
  const margins = deriveMargins(records);
  const bindings = manifestBindings({ manifestBody, manifest });
  const evidence = evidenceBindings(records);

  const lanes = [];
  for (const [lane, liveDigest] of [...liveDigests].sort(([left], [right]) => left.localeCompare(right))) {
    const backend = lane.split(":")[0];
    const provider = lane.split(":").slice(1).join(":");
    const laneBindings = bindings.filter((item) => item.lane === lane);
    const laneRecords = evidence.filter((item) => item.lane === lane);
    const bindingTally = tally(laneBindings, liveDigest);
    const recordTally = tally(laneRecords, liveDigest);
    const backendMargins = margins[backend]?.margins ?? null;
    // A lane the derivation does not model gets no invented margin: the impact terms fall back to
    // the raw counts so the lane still ranks, and the null is visible in the output.
    const staleMeasuredMargin = backendMargins?.staleMeasuredMargin ?? null;
    const weight = staleMeasuredMargin ?? 1;
    const measured = bindingTally.total + recordTally.total > 0;
    const staleCount = bindingTally.stale + recordTally.stale;
    lanes.push({
      lane,
      backend,
      provider,
      crate: declarations?.[lane]?.crate ?? null,
      liveDigest,
      liveDigestShort: liveDigest.slice(0, 12),
      capturedDigests: shortDigests([...laneBindings, ...laneRecords]),
      status: !measured
        ? "unmeasured"
        : staleCount === 0
          ? "current"
          : bindingTally.current + recordTally.current > 0
            ? "partially-stale"
            : "stale",
      models: [
        ...new Set([...laneBindings, ...laneRecords].map((item) => item.modelId).filter(Boolean)),
      ].sort(),
      bindings: { total: bindingTally.total, stale: bindingTally.stale, current: bindingTally.current },
      records: { total: recordTally.total, stale: recordTally.stale, current: recordTally.current },
      margin: backendMargins
        ? {
            staleMeasuredMargin: backendMargins.staleMeasuredMargin,
            estimateMargin: backendMargins.estimateMargin,
            hardFloor: backendMargins.hardFloor,
            source: MARGIN_SOURCE,
          }
        : null,
      impact: {
        widenedAdmissionSurface: bindingTally.stale * weight,
        widenedEvidenceSurface: recordTally.stale * weight,
      },
    });
  }

  const stale = rankLanes(lanes.filter((lane) => lane.status === "stale" || lane.status === "partially-stale"));
  return {
    generatedAgainst: {
      inferenceRevision: meta.inferenceRevision ?? null,
      digestVersion: meta.digestVersion ?? null,
      evidenceRecords: records.length,
    },
    marginSource: MARGIN_SOURCE,
    totals: {
      declaredLanes: lanes.length,
      staleLanes: stale.length,
      currentLanes: lanes.filter((lane) => lane.status === "current").length,
      unmeasuredLanes: lanes.filter((lane) => lane.status === "unmeasured").length,
      staleBindings: lanes.reduce((sum, lane) => sum + lane.bindings.stale, 0),
      staleRecords: lanes.reduce((sum, lane) => sum + lane.records.stale, 0),
    },
    staleLanes: stale.map((lane, index) => ({ rank: index + 1, ...lane })),
    currentLanes: lanes.filter((lane) => lane.status === "current"),
    unmeasuredLanes: lanes.filter((lane) => lane.status === "unmeasured"),
  };
}

export async function loadSources(root = ROOT) {
  const [closuresBody, evidenceBody, manifestBody, cargoBody] = await Promise.all(
    Object.values(SOURCE_PATHS).map((relative) => readFile(path.join(root, relative), "utf8")),
  );
  const closures = JSON.parse(closuresBody);
  return {
    // The SAME gate the matrix generator applies: a closure table not keyed to the live pin is a
    // hard error, here as there, so the report can never grade currency against a stale table.
    liveDigests: validatedInferenceClosures(closuresBody, inferencePinFromCargo(cargoBody)),
    declarations: closures.providers,
    records: JSON.parse(evidenceBody).records,
    manifest: JSON.parse(stripJsoncComments(manifestBody)),
    manifestBody,
    meta: { inferenceRevision: closures.inferenceRevision, digestVersion: closures.digestVersion },
  };
}

function percent(fraction) {
  return fraction === null || fraction === undefined ? "n/a" : `${(fraction * 100).toFixed(2)}%`;
}

export function formatReport(report) {
  const out = [];
  const meta = report.generatedAgainst;
  out.push(
    `sc-18098 stale-lane report — ${SOURCE_PATHS.closures} @ ` +
      `${meta.inferenceRevision?.slice(0, 8) ?? "(unset)"} (${meta.digestVersion ?? "unversioned"}), ` +
      `${meta.evidenceRecords} evidence records`,
  );
  out.push("");
  out.push(
    "Staleness is a SIGNAL, not a gate (epic 18093 R2/R3). A stale lane keeps serving its measured " +
      "numbers\nbehind the widened margin below; nothing in `npm run check`, `rust:check`, the " +
      "pre-push hook or CI\ndemands a re-capture. This report exists so the debt is visible on " +
      "demand instead of enforced\nper-PR.",
  );
  out.push("");
  const totals = report.totals;
  out.push(
    `${totals.declaredLanes} declared lanes: ${totals.staleLanes} stale, ${totals.currentLanes} current, ` +
      `${totals.unmeasuredLanes} unmeasured (declared but never captured).`,
  );
  out.push(
    `${totals.staleBindings} shipped calibration bindings and ${totals.staleRecords} evidence records ` +
      "are serving under a widened margin.",
  );
  out.push("");

  if (report.staleLanes.length === 0) {
    out.push("No stale lanes. Every captured lane's closure digest matches the live derivation.");
  } else {
    out.push("STALE LANES, ranked by widened admission surface (stale bindings x margin), then evidence surface:");
    out.push("");
    const header = ["#", "LANE", "BINDINGS", "RECORDS", "MARGIN", "ESTIMATE", "IMPACT", "MODELS"];
    const rows = report.staleLanes.map((lane) => [
      String(lane.rank),
      lane.lane,
      `${lane.bindings.stale}/${lane.bindings.total}`,
      `${lane.records.stale}/${lane.records.total}`,
      percent(lane.margin?.staleMeasuredMargin ?? null),
      percent(lane.margin?.estimateMargin ?? null),
      lane.impact.widenedAdmissionSurface.toFixed(3),
      lane.models.join(", ") || "(none)",
    ]);
    const widths = header.map((_, column) =>
      Math.max(header[column].length, ...rows.map((row) => row[column].length)),
    );
    const line = (row) => row.map((cell, column) => cell.padEnd(widths[column])).join("  ").trimEnd();
    out.push(line(header));
    out.push(widths.map((width) => "-".repeat(width)).join("  "));
    for (const row of rows) out.push(line(row));
    out.push("");
    for (const lane of report.staleLanes) {
      out.push(
        `  ${lane.lane}  crate=${lane.crate ?? "(undeclared)"}  live=${lane.liveDigestShort}  ` +
          `captured=${lane.capturedDigests.join(",")}  status=${lane.status}`,
      );
    }
  }

  if (report.currentLanes.length) {
    out.push("");
    out.push(
      `CURRENT (no widening applied): ${report.currentLanes.map((lane) => lane.lane).join(", ")}`,
    );
  }
  if (report.unmeasuredLanes.length) {
    out.push("");
    out.push(
      "DECLARED BUT NEVER CAPTURED (not stale — no measurement to be stale): " +
        report.unmeasuredLanes.map((lane) => lane.lane).join(", "),
    );
  }
  out.push("");
  out.push(`Margin source: ${report.marginSource}`);
  return `${out.join("\n")}\n`;
}

async function main(argv = process.argv.slice(2)) {
  const sources = await loadSources();
  const report = buildStaleLaneReport(sources);
  process.stdout.write(argv.includes("--json") ? `${JSON.stringify(report, null, 2)}\n` : formatReport(report));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main().catch((error) => {
    process.stderr.write(`${error?.stack ?? error}\n`);
    process.exitCode = 1;
  });
}
