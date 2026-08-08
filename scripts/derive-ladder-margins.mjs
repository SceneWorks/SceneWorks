#!/usr/bin/env node

/**
 * sc-18094 — derive the ladder margin policy from repeat-capture variance in the calibration
 * evidence corpus.
 *
 * Epic 18093 retires "measurement currency as a gate": stale-closure measured evidence stays
 * eligible (sc-18095) and estimate-backed candidates become admissible (sc-18096/18097), both
 * behind a WIDENED margin instead of a hard invalidation. This script is the derivation for that
 * margin. It answers one question from the artifact rather than from prose: when the harness
 * captures the SAME evidence key twice, how far apart do the phase peaks land?
 *
 * Definitions, kept strictly separate:
 *
 *   REPEAT PAIR      — two records sharing the full evidence key
 *                      (modelId/provider/mode/tier/overlay + backend + rung + strategy
 *                      parameters + geometry). Fingerprint and capture time are deliberately NOT
 *                      in the key: a re-capture under a refreshed harness fingerprint is exactly
 *                      the "same cell, different day" situation the margin must absorb.
 *   BINDING spread   — a pairwise relative spread on a phase peak that is at least
 *                      BINDING_PHASE_ENVELOPE_SHARE of the record's overall allocator envelope.
 *                      Only these can move a SAME-CELL admission decision (stale-measured,
 *                      sc-18095): the gate compares peaks against a budget, and — when admitting
 *                      the very cell that was measured — a phase far below the envelope cannot
 *                      become the envelope within the spread bounds admitted here (a
 *                      <50%-of-envelope phase with a <=50% spread stays <=75% of the envelope).
 *                      That argument does NOT extend to estimate-backed extrapolation: at a
 *                      different rung (e.g. bounded conditioning) or larger geometry (MLX
 *                      activation transients scale linearly in area) a phase that was minor in
 *                      the measured cell can carry the request peak. See CAN-BIND below.
 *   CAN-BIND spread  — any pairwise per-phase spread that is NOT an accounting flip, regardless
 *                      of the phase's share of the envelope. Under estimate-backed extrapolation
 *                      (sc-18096/18097) any phase can become the binding one, so per-phase
 *                      re-capture variance is tracked separately from the same-cell binding set
 *                      and reported as `maxCanBindPhaseSpread`.
 *   ACCOUNTING FLIP  — a pairwise spread above ACCOUNTING_DISCONTINUITY_THRESHOLD. The corpus
 *                      contains conditioning-phase peaks of ~44 KB in one harness variant and
 *                      ~16 GB in another for the same key (the eager variant reports conditioning
 *                      before weights are resident). That is a change in WHAT is measured, not
 *                      capture noise; folding it into a variance estimate would produce a
 *                      nonsense margin, so it is excluded and counted separately.
 *
 * Margin rule (per backend):
 *
 *   staleMeasuredMargin = max(hardFloor, VARIANCE_SAFETY_MULTIPLIER x maxBindingSpread)
 *   estimateMargin      = ESTIMATE_WIDENING_MULTIPLIER x staleMeasuredMargin
 *
 * The hard floor applies when variance data is thin — few or no repeat pairs — and, as of the
 * 65-record corpus, it binds for BOTH backends (see the floor rationale on each constant below).
 *
 * The estimate margin does NOT fold in the max CAN-BIND phase spread. The corpus demonstrates a
 * 17.1369% cross-fingerprint same-key denoise re-capture spread; folding it through the rule
 * shape above (x2 safety, floor, x2 widening) would yield a 68.5% MLX estimate margin —
 * unusable (and even the un-widened variance term alone is 34.3%). Instead
 * ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE records the constraint that carries that
 * risk: estimate-backed admission (sc-18096/18097) must not admit a candidate whose predicted
 * binding phase differs from the measured cell's without re-deriving per-phase variance for
 * that phase. The derivation prints the can-bind number so the tradeoff stays visible as the
 * corpus grows.
 *
 * The constants this script derives are landed in
 * `crates/sceneworks-worker/src/ladder_margin_policy.rs`;
 * `scripts/derive-ladder-margins.test.mjs` pins the two against each other, so re-running this
 * derivation after the evidence grows will red that test the moment the variance term overtakes
 * a floor.
 *
 * Run: `node scripts/derive-ladder-margins.mjs [--json]`
 */

import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const EVIDENCE_PATH = "docs/generated/memory-calibration-evidence.json";

/**
 * Hard floor for MLX, and the whole MLX margin today (the observed variance term is 2.48%, under
 * the floor). Why 5% and not the observed 1.24% max binding spread:
 *
 *   1. FAILURE COST IS ASYMMETRIC AND FATAL. An MLX allocator overshoot aborts the worker
 *      process via the default MLX error handler — there is no recoverable Err on that lane —
 *      so the margin must absorb variance we have NOT yet sampled, not just variance we have.
 *   2. THE CORPUS ITSELF DEMONSTRATES ~5% ENVELOPE HEADROOM. Across all 50 MLX records the
 *      shipped predictor's envelope gap (predicted - observed)/predicted spans 4.76%..5.21%.
 *      That is the headroom the calibrated pipeline already carries on every measured cell; a
 *      floor of 5% keeps stale/estimate admission no more aggressive than the predictor's own
 *      demonstrated gap. This is the citable anchor — the epic 15448-era ~5-6% cold-allocator
 *      settle observations are not restated anywhere greppable in docs/, so this floor is
 *      grounded in the evidence file instead.
 *   3. REPEAT COVERAGE IS THIN. 17 repeat groups spanning 2 model families on 1 machine make
 *      the 1.24% observed maximum a lower bound on true capture variance, not an estimate of it.
 */
export const MLX_HARD_FLOOR = 0.05;

/**
 * Hard floor for candle, and — by the sc-18094 rule for a backend with ZERO repeat pairs — the
 * whole candle margin: all 15 candle records are unique keys, so there is no variance estimate
 * and none is invented. Why the floor is 2.5x smaller than MLX's:
 *
 *   1. FAILURE COST IS RECOVERABLE. A CUDA/candle allocation failure surfaces as an Err the
 *      worker maps to a failed job; the process survives and the job can rerun.
 *   2. THE ACCOUNTING IS DETERMINISTIC LIVE-ALLOCATION COUNTING, not an allocator pool
 *      envelope: 10 of 15 candle records report observed == predicted to the byte
 *      (requestLiveAllocationPeak), and no candle record shows reclaimable slack.
 */
export const CANDLE_HARD_FLOOR = 0.02;

/**
 * Widening applied to the observed max binding spread before comparing with the floor. A max
 * over 39 pairs is an order statistic, not a bound; doubling it is the cheap protection against
 * the next capture landing just outside the sampled range.
 */
export const VARIANCE_SAFETY_MULTIPLIER = 2;

/**
 * Estimate-backed candidates (sc-18096/18097) carry model extrapolation error ON TOP of capture
 * variance — the estimator is projecting a cell nobody has measured — so their margin is double
 * the stale-measured margin, which only has to absorb re-capture noise on a cell that WAS
 * measured.
 */
export const ESTIMATE_WIDENING_MULTIPLIER = 2;

/**
 * Constraint inherited by sc-18096/18097, mirrored by the Rust constant of the same name (the
 * test pins the two against each other): estimate-backed admission must not admit a candidate
 * whose predicted binding phase differs from the measured cell's binding phase without
 * re-deriving per-phase variance for that phase. This is the load-bearing replacement for
 * folding `maxCanBindPhaseSpread` (17.1369% as of this corpus — larger than every margin below)
 * into the estimate margin; see the header for the arithmetic that rules the fold-in out.
 *
 * SCOPE: this constraint governs estimate candidates extrapolated from a measured cell (fitted
 * per-phase curves). Candidates with no measured cell in their extrapolation basis — the
 * weights + headroom floor path of epic 18093 R1 — have no measured binding phase to match and
 * are NOT gated by this constraint; their risk is carried by the headroom floor and the
 * estimate margin, not this rule.
 */
export const ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE = true;

/** Pairwise spreads above this are accounting flips (see header), excluded and counted. */
export const ACCOUNTING_DISCONTINUITY_THRESHOLD = 0.5;

/** A phase peak must reach this share of the record's envelope for its spread to be binding. */
export const BINDING_PHASE_ENVELOPE_SHARE = 0.5;

const PHASES = ["conditioning", "denoise", "decode"];

/** Full evidence key per the sc-18094 definition: route + backend + tier + strategy + geometry. */
export function evidenceKey(record) {
  const target = record.target ?? {};
  const strategy = record.strategy ?? {};
  const parameters = strategy.parameters ?? {};
  const sortedParameters = Object.fromEntries(
    Object.keys(parameters)
      .sort()
      .map((name) => [name, parameters[name]]),
  );
  return JSON.stringify([
    target.modelId ?? null,
    target.provider ?? null,
    target.mode ?? null,
    target.tier ?? null,
    target.overlay ?? null,
    record.backend ?? null,
    strategy.rung ?? null,
    sortedParameters,
    target.geometry ?? null,
  ]);
}

function envelopeBytes(record) {
  const overall = record.observedMemory?.overall ?? {};
  return overall.allocatorBytes ?? overall.activeBytes ?? null;
}

/**
 * Range of the shipped predictor's envelope gap `(predicted - observed) / predicted` across the
 * given records. This is the citable anchor behind the 5% MLX hard floor (rationale 2 on
 * MLX_HARD_FLOOR): the headroom the calibrated pipeline already demonstrates on every measured
 * cell. Computed here — not asserted from prose — and pinned in the test alongside the other
 * doc-fact guards.
 */
export function predictorEnvelopeGapRange(records) {
  const gaps = records
    .map((record) => {
      const predicted = record.predictedPeakBytes?.overall ?? null;
      const observed = envelopeBytes(record);
      return predicted && observed ? (predicted - observed) / predicted : null;
    })
    .filter((gap) => gap !== null);
  if (gaps.length === 0) return null;
  return { count: gaps.length, min: Math.min(...gaps), max: Math.max(...gaps) };
}

function phasePeaks(record, phase) {
  if (phase === "overall") {
    const bytes = envelopeBytes(record);
    return bytes ? [["allocatorBytes", bytes]] : [];
  }
  const observed = record.observedMemory?.[phase] ?? {};
  return ["activeBytes", "allocatorBytes"]
    .filter((metric) => observed[metric])
    .map((metric) => [metric, observed[metric]]);
}

/**
 * Analyze one backend's records: group by full evidence key, walk every within-group pair, and
 * classify each phase-peak spread as binding, non-binding, or an accounting flip.
 *
 * Also scans record diagnostics for repeated identical-request byte measurements (the sc-18094
 * requirements ask for them to be considered if present): as of the current corpus every record
 * carries exactly one measurement per phase and the scenario entries carry no re-capture bytes,
 * so `intraRecordRepeatMeasurements` is 0 and repeat PAIRS are the only variance source.
 */
export function analyzeBackend(records) {
  const groups = new Map();
  for (const record of records) {
    const key = evidenceKey(record);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(record);
  }

  const analysis = {
    recordCount: records.length,
    uniqueKeys: groups.size,
    repeatGroups: 0,
    repeatPairs: 0,
    accountingFlipsExcluded: 0,
    intraRecordRepeatMeasurements: 0,
    bindingSpreads: [],
    maxNonBindingSpread: null,
    maxCanBindPhaseSpread: null,
    envelopeGap: predictorEnvelopeGapRange(records),
  };

  for (const record of records) {
    const names = (record.diagnostics?.measurements ?? []).map((entry) => entry.name);
    analysis.intraRecordRepeatMeasurements += names.length - new Set(names).size;
  }

  for (const members of groups.values()) {
    if (members.length < 2) continue;
    analysis.repeatGroups += 1;
    const sorted = [...members].sort((a, b) => a.id.localeCompare(b.id));
    for (let i = 0; i < sorted.length; i += 1) {
      for (let j = i + 1; j < sorted.length; j += 1) {
        analysis.repeatPairs += 1;
        const [first, second] = [sorted[i], sorted[j]];
        for (const phase of [...PHASES, "overall"]) {
          const firstPeaks = new Map(phasePeaks(first, phase));
          for (const [metric, secondBytes] of phasePeaks(second, phase)) {
            const firstBytes = firstPeaks.get(metric);
            if (!firstBytes) continue;
            const spread = Math.abs(firstBytes - secondBytes) / Math.min(firstBytes, secondBytes);
            if (spread > ACCOUNTING_DISCONTINUITY_THRESHOLD) {
              analysis.accountingFlipsExcluded += 1;
              continue;
            }
            if (
              analysis.maxCanBindPhaseSpread === null ||
              spread > analysis.maxCanBindPhaseSpread.spread
            ) {
              analysis.maxCanBindPhaseSpread = { spread, phase, metric, recordIds: [first.id, second.id] };
            }
            const binding =
              phase === "overall" ||
              Math.max(
                firstBytes / envelopeBytes(first),
                secondBytes / envelopeBytes(second),
              ) >= BINDING_PHASE_ENVELOPE_SHARE;
            if (binding) {
              analysis.bindingSpreads.push({
                spread,
                phase,
                metric,
                recordIds: [first.id, second.id],
              });
            } else if (
              analysis.maxNonBindingSpread === null ||
              spread > analysis.maxNonBindingSpread.spread
            ) {
              analysis.maxNonBindingSpread = { spread, phase, metric, recordIds: [first.id, second.id] };
            }
          }
        }
      }
    }
  }

  analysis.bindingSpreads.sort((a, b) => b.spread - a.spread);
  return analysis;
}

/** Apply the margin rule (see header) to one backend's analysis. */
export function deriveBackendMargins(analysis, hardFloor) {
  const maxBindingSpread = analysis.bindingSpreads[0]?.spread ?? null;
  const varianceTerm =
    maxBindingSpread === null ? null : VARIANCE_SAFETY_MULTIPLIER * maxBindingSpread;
  const staleMeasuredMargin = Math.max(hardFloor, varianceTerm ?? 0);
  return {
    hardFloor,
    maxBindingSpread,
    varianceTerm,
    floorBinds: (varianceTerm ?? 0) <= hardFloor,
    staleMeasuredMargin,
    estimateMargin: ESTIMATE_WIDENING_MULTIPLIER * staleMeasuredMargin,
  };
}

export function deriveMargins(records) {
  const perBackend = {};
  for (const [backend, hardFloor] of [
    ["mlx", MLX_HARD_FLOOR],
    ["candle", CANDLE_HARD_FLOOR],
  ]) {
    const analysis = analyzeBackend(records.filter((record) => record.backend === backend));
    perBackend[backend] = { analysis, margins: deriveBackendMargins(analysis, hardFloor) };
  }
  return perBackend;
}

export async function loadEvidenceRecords(root = ROOT) {
  const evidence = JSON.parse(await readFile(path.join(root, EVIDENCE_PATH), "utf8"));
  return evidence.records;
}

function formatPercent(fraction) {
  return fraction === null ? "n/a" : `${(fraction * 100).toFixed(4)}%`;
}

async function main() {
  const records = await loadEvidenceRecords();
  const derived = deriveMargins(records);

  if (process.argv.includes("--json")) {
    process.stdout.write(`${JSON.stringify(derived, null, 2)}\n`);
    return;
  }

  process.stdout.write(`sc-18094 ladder margin derivation — ${EVIDENCE_PATH} (${records.length} records)\n`);
  for (const [backend, { analysis, margins }] of Object.entries(derived)) {
    process.stdout.write(`\n[${backend}] ${analysis.recordCount} records, ${analysis.uniqueKeys} unique keys\n`);
    process.stdout.write(
      `  repeat groups: ${analysis.repeatGroups}   repeat pairs: ${analysis.repeatPairs}   ` +
        `accounting flips excluded: ${analysis.accountingFlipsExcluded}   ` +
        `intra-record repeat measurements: ${analysis.intraRecordRepeatMeasurements}\n`,
    );
    const top = analysis.bindingSpreads[0];
    process.stdout.write(
      top
        ? `  max binding spread: ${formatPercent(top.spread)} (${top.phase}/${top.metric}, ${top.recordIds.join(" vs ")})\n`
        : "  max binding spread: none — no repeat pairs; the hard floor is the whole margin\n",
    );
    if (analysis.maxNonBindingSpread) {
      const nb = analysis.maxNonBindingSpread;
      process.stdout.write(
        `  max NON-binding spread (excluded from SAME-CELL margins): ${formatPercent(nb.spread)} (${nb.phase}/${nb.metric}, ${nb.recordIds.join(" vs ")})\n`,
      );
    }
    if (analysis.maxCanBindPhaseSpread) {
      const cb = analysis.maxCanBindPhaseSpread;
      const foldedIn = ESTIMATE_WIDENING_MULTIPLIER * Math.max(margins.hardFloor, VARIANCE_SAFETY_MULTIPLIER * cb.spread);
      process.stdout.write(
        `  max CAN-BIND phase spread (any phase, flips excluded): ${formatPercent(cb.spread)} (${cb.phase}/${cb.metric}, ${cb.recordIds.join(" vs ")})\n` +
          `    folding it into the estimate margin would yield ${formatPercent(foldedIn)} — carried by the phase-extrapolation constraint instead (see header)\n`,
      );
    }
    if (analysis.envelopeGap) {
      const gap = analysis.envelopeGap;
      process.stdout.write(
        `  predictor envelope gap (predicted - observed)/predicted: ${formatPercent(gap.min)}..${formatPercent(gap.max)} across ${gap.count} records\n`,
      );
    }
    process.stdout.write(
      `  variance term (x${VARIANCE_SAFETY_MULTIPLIER}): ${formatPercent(margins.varianceTerm)}   hard floor: ${formatPercent(margins.hardFloor)}   floor binds: ${margins.floorBinds}\n`,
    );
    process.stdout.write(
      `  => staleMeasuredMargin = ${formatPercent(margins.staleMeasuredMargin)}   estimateMargin = ${formatPercent(margins.estimateMargin)}\n`,
    );
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main().catch((error) => {
    process.stderr.write(`${error?.stack ?? error}\n`);
    process.exitCode = 1;
  });
}
