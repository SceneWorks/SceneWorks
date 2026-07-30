#!/usr/bin/env node

/**
 * SC-16056 — how many measurement runs does certifying all 53 catalog entries actually take?
 *
 * `docs/generated/memory-matrix.json` has thousands of cells. Read naively — one cell, one run —
 * calibration looks unaffordable. This generator refuses to answer that from prose. It reads the
 * generated matrix, derives the cell -> run collapsing rule from the harness code that actually
 * emits records, and keeps three tiers of confidence strictly separated:
 *
 *   CELLS are a FACT           — counted from the artifact.
 *   RUNS are DERIVED           — from the collapsing rule below, every axis cited to code.
 *   HOURS are PARAMETERISED    — per-run cost is an INPUT with a swept range, never an invented fact.
 *
 * Nothing here is an estimate presented as a measurement. Where a value cannot be known today
 * (the final Structurally-N/A count, per-run seconds) the model exposes it as a parameter and
 * reports the sensitivity instead of picking a number.
 */

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { REQUIRED_SCENARIOS } from "./memory-calibration-harness.mjs";
import { canonicalSourceText } from "./generate-memory-matrix.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT_JSON = "docs/generated/calibration-cost-model.json";
const OUTPUT_MD = "docs/generated/calibration-cost-model.md";

/**
 * The five ladder rungs, in the epic's normative cost order. `resident` is rung 0 and is
 * deliberately first: it is the one rung that can never be Structurally N/A, because it is the
 * unoptimised baseline every model has by definition. That makes its cell count an irreducible
 * measurement floor, which is the single most load-bearing fact in the N/A sensitivity below.
 */
const RUNG_ORDER = [
  "resident",
  "staged_residency",
  "bounded_decode",
  "bounded_attention",
  "bounded_transformer_residency",
];

const SESSION_KEY_FIELDS = ["modelId", "backend", "tier", "mode", "overlay"];
// Overlay dropped: a warm generator that can attach and detach an adapter amortises one model load
// across a target's overlays. Whether that is legitimate is a PROVIDER fact, not an artifact fact —
// a LoRA is cheap to swap, an identity overlay loads extra encoder weights and is not — so both
// keyings are reported and neither is asserted.
const OVERLAY_AMORTISED_SESSION_KEY_FIELDS = ["modelId", "backend", "tier", "mode"];
const FIT_GROUP_FIELDS = ["modelId", "backend", "mode", "overlay"];

function sha256(body) {
  return createHash("sha256").update(body).digest("hex");
}

// Joined on an escaped NUL rather than a raw one: a literal NUL byte in the source makes git
// classify this file as binary, which silently costs every future reader a reviewable diff. The
// separator itself must stay collision-proof because these keys are compared for identity.
function keyOf(cell, fields) {
  return fields.map((field) => cell[field]).join("\u0000");
}

function tally(values) {
  const counts = new Map();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return Object.fromEntries([...counts.entries()].sort(([left], [right]) => left.localeCompare(right)));
}

/**
 * Deterministic rounding. Every derived number in the artifact goes through this so a re-run is
 * byte-identical regardless of how the platform prints the tail of a double.
 */
function round(value, places = 2) {
  if (!Number.isFinite(value)) return null;
  const factor = 10 ** places;
  return Math.round(value * factor + Number.EPSILON * factor) / factor;
}

/**
 * PARAMETERS. Every one of these is a value the model cannot derive from the artifact today.
 * Each carries its own provenance string, and each is swept rather than trusted.
 *
 * `derivedFrom: null` means ASSUMPTION — nothing in the repository establishes it.
 */
export const DEFAULT_PARAMETERS = {
  // Structurally N/A cells need NO measurement: the epic accepts static architecture evidence
  // where the impossibility is structural. SC-15969 (the per-family rung-4 applicability survey)
  // has NOT run, so the final count is unknowable. Expressed as a per-rung fraction and swept.
  structurallyNotApplicableFractionByRung: {
    resident: 0,
    staged_residency: 0,
    bounded_decode: 0,
    bounded_attention: 0,
    bounded_transformer_residency: 0,
  },
  // Seconds per provider invocation. The measured anchors below are a FLOOR that is known not to
  // cover the dominant cost, so this default is an ASSUMPTION and the sweep is the real answer.
  perRunSeconds: 300,
  perRunSecondsSweep: [30, 60, 120, 300, 600, 1200],
  // Geometry points needed to determine one affine phase curve `fixedGb + perMpxGb * megapixels`.
  // Two is the algebraic minimum for a line. See `krea_phase_curve` in vram_gate.rs.
  geometryPointsPerFit: 2,
  // Whether the geometry slope may be shared across tiers so only one tier needs two points.
  // Default false: the shipped Krea evidence CONTRADICTS it on the decode phase (see the
  // precedent block below), and assuming true is exactly the kind of comfortable shortcut this
  // model exists to refuse.
  slopeSharedAcrossTiers: false,
  // Fraction of cells spot-checked against the fitted curve under a fit-then-validate strategy.
  validationSampleFraction: 0.05,
};

/**
 * THE COLLAPSING RULE.
 *
 * Derived by reading `scripts/memory-calibration-harness.mjs` and
 * `scripts/generate-memory-matrix.mjs`, not by assuming. Four candidate axes were tested; two
 * collapse, two do not. The two negative results are as load-bearing as the positive ones, so
 * they are recorded here rather than omitted.
 */
export const COLLAPSING_AXES = [
  {
    axis: "geometry",
    verdict: "collapsed",
    factor: "already applied by the matrix generator",
    rule:
      "A cell carries a geometry ENVELOPE, not a geometry. One record binds the whole envelope: " +
      "`calibrationBinding` accepts any record whose `width x height` appears in " +
      "`cell.geometryEnvelope.resolutions`, and only rejects batch/frames != 1.",
    citation: "scripts/generate-memory-matrix.mjs#calibrationBinding (resolution/batch/frames checks)",
    caveat:
      "This is ACCEPTANCE, not evidence. The manifest states the opposite policy in its own words " +
      "— evidenceRecords are 'exact request envelopes, not permission to interpolate: a geometry " +
      "absent here is Implemented/unverified even when a phase curve can predict it'. The binding " +
      "logic is the more permissive of the two, so this collapse is only sound if the envelope " +
      "claim is backed by a fitted curve rather than by one measured point.",
  },
  {
    axis: "strategyParameters",
    verdict: "collapsed",
    factor: "one record retires every passed sweep point",
    rule:
      "One `complete` record carries a `sweep` whose passed cases each retire a distinct planned " +
      "logical case, so a provider that sweeps its parameter domain internally needs one record " +
      "for the whole domain rather than one per point.",
    citation:
      "scripts/memory-calibration-harness.mjs#completedLogicalIds (maps every passed sweep case " +
      "to its own logicalCaseId) and #validateComplete (every declared axis range must be derived " +
      "from passed executed cases)",
    caveat:
      "Realised only ACROSS resumed invocations. `runProviderPlan` computes its skip set once from " +
      "`--resume` and then spawns the provider command once per planned case, without feeding " +
      "incoming records back into the skip set — so within a single invocation an N-point sweep " +
      "still costs N process spawns.",
  },
  {
    axis: "rung",
    verdict: "not collapsed by the harness; collapsible in principle",
    factor: "5 (every session key spans exactly 5 rungs)",
    rule:
      "A record names ONE rung (`strategy.rung` is a single enum and is part of the record " +
      "identity), and the matrix binds a record to exactly one cell on the " +
      "(modelId, provider, backend, tier, mode, overlay, rung) key. So the schema forces one " +
      "record per rung. The PHYSICS does not: the shipped Krea evidence records carry " +
      "`observedPeaksGb` for all four constrained rungs from a single capture, and the Candle " +
      "adapter is documented as running Resident -> BoundedTransformerResidency -> Resident on " +
      "'one loaded generator'.",
    citation:
      "scripts/memory-calibration-harness.mjs#recordId + #validateRecord (rung is in the identity); " +
      "scripts/generate-memory-matrix.mjs (record->cell match includes `rung`); " +
      "config/manifests/builtin.models.jsonc#krea_2_turbo/candle/turboFit/evidenceRecords " +
      "(one record, four rung peaks); docs/memory-calibration-harness.md (Candle adapter, " +
      "'one loaded generator')",
    caveat:
      "The harness spawns a FRESH process per planned case, so today the model load is paid once " +
      "per rung. Capturing this 5x needs either a warm provider daemon behind the argv command or " +
      "a runner that batches a target's rungs into one invocation. Neither exists.",
  },
  {
    axis: "overlay",
    verdict: "not collapsed for records; collapsible for model loads",
    factor: "2.47 (1,319 overlay-keyed session keys over 534 overlay-amortised ones)",
    rule:
      "3,925 of the 6,595 cells carry a non-`none` overlay, so this axis is 59.5% of the matrix. " +
      "It does NOT collapse on the record axis: `overlay` is part of the cell key and of " +
      "`record.target`, and `validateComplete` requires the `overlay` scenario to actually pass " +
      "(or carry a justified `not_applicable` reason) — a `none` record cannot certify a `lora` " +
      "cell. It DOES collapse on the model-load axis for any overlay a warm generator can attach " +
      "and detach without reloading the base weights.",
    citation:
      "docs/generated/memory-matrix.json (overlay distribution); " +
      "scripts/memory-calibration-harness.mjs#validateRecord (target.overlay is required) and " +
      "#validateComplete (the overlay scenario must pass or be justified)",
    caveat:
      "The load-side collapse is NOT uniform and is not asserted here. A LoRA is cheap to swap; an " +
      "identity overlay (InstantID / PuLID) materialises extra encoder weights and is not free. " +
      "Which overlays are load-cheap is a provider fact this artifact cannot establish, so both " +
      "keyings are reported. The 3,925 records do not evaporate under either.",
  },
  {
    axis: "cell",
    verdict: "NOT collapsed",
    factor: "1 (cells map 1:1 to certifying records)",
    rule:
      "No record can certify two cells. The matrix matches a record to a cell on the full " +
      "(modelId, provider, backend, tier, mode, overlay, rung) key, and a `complete` record must " +
      "additionally carry its own measured negative mutation — so the negative case is INSIDE the " +
      "positive record rather than being a second record.",
    citation:
      "scripts/generate-memory-matrix.mjs (record->cell match on the full key); " +
      "scripts/memory-calibration-harness.mjs#validateComplete (negativeMutation must be measured " +
      "and fail as expected inside the complete record)",
    caveat:
      "This is the negative result. Tier, mode and overlay are all part of the cell key and none " +
      "of them collapses: a q4 record cannot certify a q8 cell, and a text_to_image record cannot " +
      "certify an edit_image cell.",
  },
];

/**
 * PRODUCIBILITY. A cell nobody can currently measure is a different cost category from a cell that
 * is merely unmeasured, and conflating them is what makes a run count look like a GPU-hours problem
 * when it is a code problem.
 *
 * The gates are ORDERED and a cell is charged to the FIRST one that blocks it, so the buckets
 * partition the matrix exactly. Every gate is derived from a file that was read, not assumed.
 */
export const PRODUCIBILITY_GATES = [
  {
    id: "no-run-needed",
    label: "Structurally N/A — static architecture evidence is sufficient",
    blockedBy: "nothing; the epic exempts these",
    citation: "docs/generated/memory-matrix.json (cell.state) and the epic's conformance states",
  },
  {
    id: "overlay-declined",
    label: "Non-`none` overlay: no adapter can emit a truthful overlay scenario",
    blockedBy: "provider-adapter code",
    citation:
      "crates/sceneworks-memory-adapter/src/bin/candle.rs hardcodes the overlay scenario to " +
      '`not_applicable` with the fixed reason "ordinary Krea Turbo text-to-image calibration has ' +
      'no overlay", which is false for a lora/identity/control target; ' +
      "crates/sceneworks-memory-adapter/src/bin/mlx.rs contains no overlay handling at all, so its " +
      "overlay scenario stays `not_run` (from lib.rs#not_run_scenarios), which " +
      "scripts/memory-calibration-harness.mjs#validateComplete rejects outright",
  },
  {
    id: "no-provider-adapter",
    label: "No provider adapter covers this (entry, backend) at all",
    blockedBy: "provider-adapter code",
    citation: "config/memory-calibration-plan.json (the shipped plan's provider targets)",
  },
  {
    id: "adapter-gated",
    label: "An adapter exists, but it has never emitted a `complete` record",
    blockedBy: "provider-adapter code (phase telemetry and lifecycle injection)",
    citation:
      "docs/generated/memory-calibration-evidence.json (records with status `complete`, per " +
      "(entry, backend)); docs/memory-calibration-harness.md documents both adapters as returning " +
      "`gated` records today",
  },
  {
    id: "producible-today",
    label: "A run executed right now could certify this cell",
    blockedBy: "nothing — GPU hours only",
    citation: "the remainder after the gates above",
  },
];

export function producibility(cells, { adapterPairs, pairsWithCompleteRecords }) {
  const adapters = new Set(adapterPairs);
  const complete = new Set(pairsWithCompleteRecords);
  const buckets = new Map(PRODUCIBILITY_GATES.map((gate) => [gate.id, []]));

  for (const cell of cells) {
    const pair = `${cell.modelId}|${cell.backend}`;
    const gate =
      cell.state === "Structurally N/A"
        ? "no-run-needed"
        : cell.overlay !== "none"
          ? "overlay-declined"
          : !adapters.has(pair)
            ? "no-provider-adapter"
            : !complete.has(pair)
              ? "adapter-gated"
              : "producible-today";
    buckets.get(gate).push(cell);
  }

  const partition = PRODUCIBILITY_GATES.map((gate) => {
    const bucket = buckets.get(gate.id);
    return {
      ...gate,
      cells: bucket.length,
      mlx: bucket.filter((cell) => cell.backend === "mlx").length,
      candle: bucket.filter((cell) => cell.backend === "candle").length,
    };
  });

  const total = cells.length;
  const producible = buckets.get("producible-today").length;
  const noRunNeeded = buckets.get("no-run-needed").length;
  const behindCode = total - producible - noRunNeeded;

  return {
    partition,
    // The number that reframes the whole model: how much of the campaign is gated on code that does
    // not exist, rather than on machine time that could start today.
    behindProviderCode: behindCode,
    behindProviderCodeShare: round(behindCode / total, 4),
    producibleToday: producible,
    headline:
      producible === 0
        ? "ZERO cells are producible today. The binding constraint on this epic is provider-adapter " +
          "code, not GPU hours: no adapter has ever emitted a `complete` record, and 2 of 53 " +
          "entries have an adapter at all. Every hour in the wall-clock sweep below is hypothetical " +
          "until that changes."
        : `${producible} of ${total} cells could be certified by a run started right now.`,
  };
}

/** Layer 1 — CELLS. Pure counting off the artifact. No interpretation. */
export function cellCensus(cells) {
  const sessions = new Map();
  const fitGroups = new Map();
  let geometryExpandedCells = 0;

  for (const cell of cells) {
    const resolutions = cell.geometryEnvelope?.resolutions ?? [];
    const points = Math.max(resolutions.length, 1);
    geometryExpandedCells += points;

    const sessionKey = keyOf(cell, SESSION_KEY_FIELDS);
    if (!sessions.has(sessionKey)) {
      sessions.set(sessionKey, { backend: cell.backend, geometryPoints: points, rungs: new Set() });
    }
    sessions.get(sessionKey).rungs.add(cell.rung);

    const fitKey = keyOf(cell, FIT_GROUP_FIELDS);
    if (!fitGroups.has(fitKey)) fitGroups.set(fitKey, { backend: cell.backend, tiers: new Set() });
    fitGroups.get(fitKey).tiers.add(cell.tier);
  }

  const sessionList = [...sessions.values()];
  const rungSpans = new Set(sessionList.map((session) => session.rungs.size));

  return {
    total: cells.length,
    byState: tally(cells.map((cell) => cell.state)),
    byBackend: tally(cells.map((cell) => cell.backend)),
    byStateAndBackend: tally(cells.map((cell) => `${cell.state} @ ${cell.backend}`)),
    byRung: tally(cells.map((cell) => cell.rung)),
    byTier: tally(cells.map((cell) => cell.tier)),
    byMode: tally(cells.map((cell) => cell.mode)),
    byOverlay: tally(cells.map((cell) => cell.overlay)),
    // Geometry is the collapse the matrix ALREADY made. Keyed per resolution the matrix would be
    // this large; the ratio is how much collapsing is baked into the published cell count.
    geometryExpandedCells,
    geometryCollapseFactor: round(geometryExpandedCells / cells.length, 3),
    sessions: {
      total: sessions.size,
      byBackend: tally(sessionList.map((session) => session.backend)),
      geometryExpanded: sessionList.reduce((sum, session) => sum + session.geometryPoints, 0),
      // If this is not exactly [5] the rung collapse factor below is not a constant and the
      // model must not report one.
      rungsPerSession: [...rungSpans].sort((left, right) => left - right),
    },
    fitGroups: {
      total: fitGroups.size,
      byBackend: tally([...fitGroups.values()].map((group) => group.backend)),
      tierSum: [...fitGroups.values()].reduce((sum, group) => sum + group.tiers.size, 0),
    },
  };
}

/**
 * Layer 2 — RUNS. Applies the collapsing rule and the N/A exemption to produce run counts under
 * each realisable strategy, split by hardware because MLX/Metal runs on a Mac and Candle/CUDA
 * needs rented CUDA — completely different cost structures.
 */
export function collapseToRuns(cells, parameters = DEFAULT_PARAMETERS) {
  const naFractions = {
    ...DEFAULT_PARAMETERS.structurallyNotApplicableFractionByRung,
    ...(parameters.structurallyNotApplicableFractionByRung ?? {}),
  };

  // Already-classified N/A cells are exempt as a FACT. Additional exemptions are a PROJECTION.
  // The projection is applied to NAMED cells, not to a count, so the session arithmetic below
  // stays exact instead of becoming a division. Selection is by sorted cell id: deterministic,
  // and arbitrary in a way the output says out loud rather than hiding behind a percentage.
  const measurable = cells.filter((cell) => cell.state !== "Structurally N/A");
  const alreadyExempt = cells.length - measurable.length;

  const projectedExemptByRung = {};
  const projected = new Set();
  for (const rung of RUNG_ORDER) {
    const inRung = measurable
      .filter((cell) => cell.rung === rung)
      .sort((left, right) => left.id.localeCompare(right.id));
    const fraction = Math.min(Math.max(naFractions[rung] ?? 0, 0), 1);
    const exempt = Math.round(inRung.length * fraction);
    projectedExemptByRung[rung] = exempt;
    for (const cell of inRung.slice(0, exempt)) projected.add(cell.id);
  }

  const surviving = measurable.filter((cell) => !projected.has(cell.id));
  const certifyingRecords = {
    total: surviving.length,
    mlx: surviving.filter((cell) => cell.backend === "mlx").length,
    candle: surviving.filter((cell) => cell.backend === "candle").length,
  };

  // A warm session is one MODEL LOAD. It is NOT the record count divided by the rung span: a
  // session survives as long as ANY of its rungs still needs measuring, so exempting rungs
  // removes records without removing loads. Counting the surviving session keys directly is what
  // makes that visible; dividing would have hidden it.
  const sessionsKeyedBy = (fields) => {
    const keys = new Map();
    for (const cell of surviving) {
      const key = keyOf(cell, fields);
      if (!keys.has(key)) keys.set(key, cell.backend);
    }
    const backends = [...keys.values()];
    return {
      total: keys.size,
      mlx: backends.filter((backend) => backend === "mlx").length,
      candle: backends.filter((backend) => backend === "candle").length,
    };
  };
  const warmSessions = sessionsKeyedBy(SESSION_KEY_FIELDS);
  const overlayAmortisedSessions = sessionsKeyedBy(OVERLAY_AMORTISED_SESSION_KEY_FIELDS);

  const census = cellCensus(cells);
  const rungCollapse =
    census.sessions.rungsPerSession.length === 1 ? census.sessions.rungsPerSession[0] : null;

  return {
    exemptions: {
      alreadyStructurallyNotApplicable: alreadyExempt,
      projectedAdditional: projected.size,
      projectedAdditionalByRung: projectedExemptByRung,
    },
    // The unit the harness actually spawns: one `action:"run"` provider invocation.
    certifyingRecords,
    // The unit that costs a model load, IF a provider amortises one across a target's rungs.
    warmSessions,
    // The same unit if the provider ALSO swaps overlays without reloading base weights. Reported
    // rather than chosen: legitimate for a LoRA, not for an identity overlay that materialises
    // extra encoder weights.
    overlayAmortisedSessions,
    overlayLoadCollapseFactor:
      overlayAmortisedSessions.total === 0
        ? null
        : round(warmSessions.total / overlayAmortisedSessions.total, 2),
    // Three places deliberately: the exempt cells make this NOT exactly the rung span, and a
    // two-place round would print a clean "5" that reads as an identity rather than a ratio.
    recordsPerWarmSession:
      warmSessions.total === 0 ? null : round(certifyingRecords.total / warmSessions.total, 3),
    rungCollapseFactor: rungCollapse,
  };
}

/**
 * N/A sensitivity. SC-15969 has not run, so the honest output is a curve, not a number.
 * The scenarios below are labelled projections; only `current` is a fact.
 */
export function naSensitivity(cells) {
  const scenarios = [
    {
      name: "current",
      kind: "fact",
      note: "What the generated matrix classifies today.",
      fractions: {},
    },
    {
      name: "rung4-half",
      kind: "projection",
      note:
        "Half of all bounded_transformer_residency cells turn out structurally inapplicable. " +
        "SC-15969 names the two breakers: heterogeneous blocks (the floor is the largest single " +
        "block, so the saving collapses) and skip connections (encoder activations must persist).",
      fractions: { bounded_transformer_residency: 0.5 },
    },
    {
      name: "rung4-all",
      kind: "projection",
      note:
        "Upper bound on the rung-4 axis alone: every rung-4 cell is exempt. This is the most " +
        "optimistic single-rung outcome and it still leaves four rungs to measure.",
      fractions: { bounded_transformer_residency: 1 },
    },
    {
      name: "rungs-2-3-4-half",
      kind: "projection",
      note:
        "Half of bounded_decode, bounded_attention and bounded_transformer_residency exempt — " +
        "i.e. a large fraction of the catalog turns out to have a non-tileable decoder AND " +
        "non-chunkable attention. Aggressive; included to show the shape of the curve, not " +
        "because anything supports it.",
      fractions: { bounded_decode: 0.5, bounded_attention: 0.5, bounded_transformer_residency: 0.5 },
    },
    {
      name: "everything-but-baseline",
      kind: "bound",
      note:
        "Absolute floor. Every constrained rung on every entry is exempt; only the resident " +
        "baseline is measured. Rung 0 can never be Structurally N/A — it is the unoptimised " +
        "baseline every model has by definition — so this is the smallest campaign that can exist.",
      fractions: {
        staged_residency: 1,
        bounded_decode: 1,
        bounded_attention: 1,
        bounded_transformer_residency: 1,
      },
    },
  ];

  return scenarios.map((scenario) => {
    const runs = collapseToRuns(cells, {
      ...DEFAULT_PARAMETERS,
      structurallyNotApplicableFractionByRung: {
        ...DEFAULT_PARAMETERS.structurallyNotApplicableFractionByRung,
        ...scenario.fractions,
      },
    });
    return {
      name: scenario.name,
      kind: scenario.kind,
      note: scenario.note,
      certifyingRecords: runs.certifyingRecords,
      warmSessions: runs.warmSessions,
    };
  });
}

/**
 * Wall clock. Per-run seconds is an INPUT. The measured anchors are recorded with exactly what
 * they do and do not cover, and the default is labelled an assumption.
 */
export function wallClock(runs, parameters = DEFAULT_PARAMETERS) {
  const sweep = [...new Set([...(parameters.perRunSecondsSweep ?? []), parameters.perRunSeconds])]
    .filter((value) => Number.isFinite(value) && value > 0)
    .sort((left, right) => left - right);

  const hours = (count, seconds) =>
    count === null ? null : round((count * seconds) / 3600, 1);

  return {
    measuredAnchors: [
      {
        what:
          "8 real MLX provider invocations plus one hardware probe and a schema check, on an " +
          "Apple M5 Max self-hosted runner.",
        measurement: "30.76 s wall clock for the whole harness step",
        derivation:
          "GitHub Actions run 30380114921 job 90347750988: cargo 'Finished `release` profile' at " +
          "17:14:45.405Z, next step group (artifact upload) at 17:15:16.164Z. The 8 records are " +
          "the ones the epic ledger describes (7 passing tile points + 1 negative mutation).",
        perInvocationSeconds: round(30.76 / 8, 2),
        doesNotCover:
          "The Qwen MLX adapter exercises the VAE decode seam ONLY. Its records are `gated`, not " +
          "`complete`: no text-encoder or DiT load, no conditioning, no denoise steps, no " +
          "cancel/error injection at every phase, no warm A->B->A repeat. A `complete` record " +
          `needs all ${REQUIRED_SCENARIOS.length} required scenarios plus quality and a measured ` +
          "negative mutation. This is a FLOOR that is known not to cover the dominant cost.",
      },
      {
        what: "Warm-page-cache sequential read of two real safetensors shards on this Mac.",
        measurement: "11.3 GB/s and 12.3 GB/s (5.03 GB and 3.99 GB reads via dd)",
        derivation:
          "dd bs=4m over Z-Image-Turbo transformer shard 1 of 3 and text_encoder shard 2 of 3 in " +
          "the local Hugging Face cache.",
        perInvocationSeconds: null,
        doesNotCover:
          "Page cache was warm and could not be dropped without elevated privileges, so this is a " +
          "best case, not cold NVMe. It bounds ONLY the file-I/O component of a model load: at " +
          "this rate a 20 GiB tier streams in under 2 s, which is the useful conclusion — I/O is " +
          "not the bottleneck. Dequantisation, graph construction and denoise are unmeasured.",
      },
    ],
    assumption: {
      parameter: "perRunSeconds",
      value: parameters.perRunSeconds,
      status: "ASSUMPTION — nothing in this repository establishes it",
      reasoning:
        "A `complete` record requires at least the 8 required scenarios, a parameter sweep, an " +
        "identical-latent quality comparison and a measured negative mutation, each involving at " +
        "least one render, on top of one model load. The default sits mid-sweep so no reader " +
        "mistakes it for a measurement. The sweep, not the default, is the answer.",
    },
    sweep: sweep.map((seconds) => ({
      perRunSeconds: seconds,
      recordHours: {
        total: hours(runs.certifyingRecords.total, seconds),
        mlx: hours(runs.certifyingRecords.mlx, seconds),
        candle: hours(runs.certifyingRecords.candle, seconds),
      },
      warmSessionHours: {
        total: hours(runs.warmSessions.total, seconds),
        mlx: hours(runs.warmSessions.mlx, seconds),
        candle: hours(runs.warmSessions.candle, seconds),
      },
      // The cheapest shape that is even arguable: rungs AND overlays amortised into one load.
      // Neither collapse exists in the harness today, so this column is a floor on a hypothetical.
      overlayAmortisedSessionHours: {
        total: hours(runs.overlayAmortisedSessions.total, seconds),
        mlx: hours(runs.overlayAmortisedSessions.mlx, seconds),
        candle: hours(runs.overlayAmortisedSessions.candle, seconds),
      },
    })),
  };
}

/**
 * Fit-versus-exhaustive. The epic splits ownership: the provider owns the FORMULA KIND, the
 * manifest owns the MEASURED COEFFICIENTS. The shipped phase curve is affine —
 * `fixedGb + perMpxGb * megapixels` — so two coefficients per (tier, rung, phase), and two
 * geometry points determine a line.
 *
 * `kreaPrecedent` is parsed from the manifest rather than quoted, so it goes stale loudly.
 */
export function fitVersusExhaustive(cells, kreaPrecedent, parameters = DEFAULT_PARAMETERS) {
  const census = cellCensus(cells);
  const runs = collapseToRuns(cells, parameters);
  const points = Math.max(1, Math.trunc(parameters.geometryPointsPerFit ?? 2));
  // Under the harness as written every rung costs its own provider invocation, so a session-level
  // count must be restated in invocations before it can be compared to a wall-clock budget.
  const rungSpan = runs.rungCollapseFactor ?? 1;

  // Exhaustive at the exact geometry the manifest's own policy demands ("not permission to
  // interpolate"): one session per (session key, resolution).
  const exhaustiveExactGeometry = census.sessions.geometryExpanded;
  // Exhaustive as the binding logic actually accepts it: one session per session key, one
  // geometry each.
  const exhaustivePerCell = census.sessions.total;
  // Fit: `points` geometries per tier, or `points` on one tier plus one on each other tier when
  // the slope may be shared. `tierSum` is the number of (group, tier) pairs, which equals the
  // session count; `fitGroups.total` is the number of groups needing the extra slope point.
  const fitSessions = parameters.slopeSharedAcrossTiers
    ? census.fitGroups.tierSum + census.fitGroups.total * (points - 1)
    : census.fitGroups.tierSum * points;
  const validationSessions = Math.ceil(
    census.sessions.total * Math.min(Math.max(parameters.validationSampleFraction ?? 0, 0), 1),
  );

  return {
    formula: {
      kind: "affine in megapixels",
      expression: "fixedGb + perMpxGb * megapixels",
      citation: "crates/sceneworks-worker/src/vram_gate.rs#krea_phase_curve",
      coefficientsPerCurve: 2,
      minimumPointsToDetermine: 2,
    },
    kreaPrecedent,
    strategies: [
      {
        name: "exhaustive-exact-geometry",
        sessions: exhaustiveExactGeometry,
        providerInvocations: exhaustiveExactGeometry * rungSpan,
        policy:
          "Measure every advertised resolution. This is what the manifest's own evidenceRecords " +
          "comment demands: 'exact request envelopes, not permission to interpolate'.",
      },
      {
        name: "exhaustive-per-cell",
        sessions: exhaustivePerCell,
        providerInvocations: runs.certifyingRecords.total,
        policy:
          "One measurement per cell at one geometry, relying on the binding logic accepting any " +
          "resolution inside the envelope. Cheaper, but it certifies an envelope from a point.",
      },
      {
        name: "fit-then-validate",
        sessions: fitSessions + validationSessions,
        providerInvocations: (fitSessions + validationSessions) * rungSpan,
        fitSessions,
        validationSessions,
        policy:
          `Measure ${points} geometry points per (entry, backend, mode, overlay, tier) to determine ` +
          "the affine curve, then spot-check a sample against the prediction. Covers every " +
          "geometry in the envelope, including ones never rendered.",
      },
    ],
    ratios: {
      exactGeometryOverFit: round(exhaustiveExactGeometry / (fitSessions + validationSessions), 2),
      exactGeometryOverPerCell: round(exhaustiveExactGeometry / exhaustivePerCell, 2),
      fitOverPerCell: round((fitSessions + validationSessions) / exhaustivePerCell, 2),
    },
    finding:
      "Fitting does NOT beat the per-cell campaign on run count — it costs more, because the " +
      "matrix already collapsed geometry and a curve needs a second geometry point the per-cell " +
      "campaign never takes. Fitting wins only against the policy the manifest actually states " +
      "(no interpolation), where it is the difference between a campaign that covers every " +
      "advertised resolution and one that does not. The decision is therefore about which " +
      "geometry policy is true, not about which strategy is cheaper.",
  };
}

/** Parse the Krea turboFit block. Derived from the manifest so the precedent cannot go stale silently. */
export function kreaFitPrecedent(manifestJson) {
  const krea = manifestJson.models.find((model) => model.id === "krea_2_turbo");
  const turboFit = krea?.candle?.turboFit;
  if (!turboFit) throw new Error("could not locate krea_2_turbo candle.turboFit in the manifest");
  const records = turboFit.evidenceRecords ?? [];
  const curves = turboFit.phaseCurvesByTier ?? {};

  const geometriesByTier = {};
  for (const record of records) {
    const tier = record.tier;
    geometriesByTier[tier] = [...new Set([...(geometriesByTier[tier] ?? []), `${record.width}x${record.height}`])].sort();
  }

  // A zero `perMpxGb` on the `text` phase is CORRECT — conditioning does not scale with output
  // area. So the geometry-sensitive phases are counted separately; a flat denoise or decode slope
  // is the one that signals an undetermined fit rather than a real physical constant.
  const geometrySensitivePhases = ["denoise", "decode"];
  let curveCount = 0;
  const slopesByTier = {};
  for (const [tier, rungs] of Object.entries(curves)) {
    let flat = 0;
    let total = 0;
    for (const phases of Object.values(rungs)) {
      for (const [phaseName, phase] of Object.entries(phases)) {
        curveCount += 1;
        if (!geometrySensitivePhases.includes(phaseName)) continue;
        total += 1;
        if (phase.perMpxGb === 0) flat += 1;
      }
    }
    slopesByTier[tier] = {
      geometryPoints: (geometriesByTier[tier] ?? []).length,
      geometrySensitiveCurves: total,
      flatSlopes: flat,
    };
  }

  const underDetermined = Object.entries(geometriesByTier)
    .filter(([, geometries]) => geometries.length < 2)
    .map(([tier]) => tier)
    .sort();
  const detail = Object.entries(slopesByTier)
    .sort()
    .map(
      ([tier, stats]) =>
        `${tier} (${stats.geometryPoints} geometry point${stats.geometryPoints === 1 ? "" : "s"}): ` +
        `${stats.flatSlopes}/${stats.geometrySensitiveCurves} flat`,
    )
    .join("; ");

  return {
    source: "config/manifests/builtin.models.jsonc#models/krea_2_turbo/candle/turboFit",
    measurementRecords: records.length,
    geometriesByTier: Object.fromEntries(Object.entries(geometriesByTier).sort()),
    tiers: Object.keys(curves).sort(),
    curves: curveCount,
    coefficients: curveCount * 2,
    slopesByTier: Object.fromEntries(Object.entries(slopesByTier).sort()),
    underDeterminedTiers: underDetermined,
    honestyNote:
      underDetermined.length === 0
        ? "Every tier carries at least two geometry points, so every slope is determined."
        : `Tiers ${underDetermined.join(", ")} carry ONE geometry point each, so their perMpxGb ` +
          "slopes cannot have been fitted. The zero-slope counts on the geometry-SENSITIVE phases " +
          `(denoise, decode) show the consequence — ${detail}. A campaign budgeted from this ` +
          "precedent must budget two geometry points per tier, not the four records this evidence " +
          "actually cost.",
  };
}

export async function buildCostModel() {
  const sourcePaths = {
    matrix: "docs/generated/memory-matrix.json",
    manifest: "config/manifests/builtin.models.jsonc",
    harness: "scripts/memory-calibration-harness.mjs",
    matrixGenerator: "scripts/generate-memory-matrix.mjs",
    calibrationPlan: "config/memory-calibration-plan.json",
    vramGate: "crates/sceneworks-worker/src/vram_gate.rs",
    calibrationEvidence: "docs/generated/memory-calibration-evidence.json",
    candleAdapter: "crates/sceneworks-memory-adapter/src/bin/candle.rs",
    mlxAdapter: "crates/sceneworks-memory-adapter/src/bin/mlx.rs",
  };
  const sourceEntries = Object.entries(sourcePaths);
  const bodies = Object.fromEntries(
    await Promise.all(
      sourceEntries.map(async ([name, relative]) => [
        name,
        canonicalSourceText(await readFile(path.join(ROOT, relative), "utf8")),
      ]),
    ),
  );

  const matrix = JSON.parse(bodies.matrix);
  const manifest = JSON.parse(stripJsoncComments(bodies.manifest));
  const plan = JSON.parse(bodies.calibrationPlan);
  const evidence = JSON.parse(bodies.calibrationEvidence);

  const cells = matrix.cells;
  const census = cellCensus(cells);
  const runs = collapseToRuns(cells, DEFAULT_PARAMETERS);
  const precedent = kreaFitPrecedent(manifest);

  // There is no completed-work baseline to subtract. Derived rather than assumed, from two
  // independent places that would have to agree: the committed evidence bundle and the matrix's own
  // summary. If either ever becomes nonzero, the run counts below stop being the whole population.
  const recordsByStatus = tally(evidence.records.map((record) => record.status));
  const completedBaseline = {
    evidenceBundleRecords: evidence.records.length,
    recordsByStatus,
    completeRecords: recordsByStatus.complete ?? 0,
    matrixSummaryCalibrationRuns: matrix.summary.calibrationRuns,
    matrixSummaryCurrentCalibrationRuns: matrix.summary.currentCalibrationRuns,
    verifiedCells: census.byState.Verified ?? 0,
    note:
      "Every run count in this document is the WHOLE POPULATION, not a remainder. Zero calibration " +
      "records exist on either backend, zero cells are `Verified`, and both the evidence bundle and " +
      "the matrix summary agree on that independently.",
  };

  // Producibility. Adapter coverage is derived from the shipped plan's provider targets; the
  // complete-record set is derived from the evidence bundle, so both self-update.
  const adapterPairs = plan.providers.map(
    (provider) => `${provider.target.modelId}|${provider.backend}`,
  );
  const pairsWithCompleteRecords = evidence.records
    .filter((record) => record.status === "complete")
    .map((record) => `${record.target.modelId}|${record.backend}`);
  const producible = producibility(cells, { adapterPairs, pairsWithCompleteRecords });

  // Control-overlay undercount. `overlaysFor` in the matrix generator emits `control` only when
  // `model[backend].control` exists, and the manifest declares it exactly once — so a shipping MLX
  // control lane produces zero cells. This is a manifest omission, and it moves the denominator.
  const controlCells = cells.filter((cell) => cell.overlay === "control");
  const controlPairs = [...new Set(controlCells.map((cell) => `${cell.modelId}|${cell.backend}`))].sort();
  const cellsPerPairIfControlDeclared = Object.fromEntries(
    [...new Set(cells.map((cell) => `${cell.modelId}|${cell.backend}`))].sort().map((pair) => {
      const inPair = cells.filter((cell) => `${cell.modelId}|${cell.backend}` === pair);
      const tiers = new Set(inPair.map((cell) => cell.tier)).size;
      const modes = new Set(inPair.map((cell) => cell.mode)).size;
      return [pair, tiers * modes * (runs.rungCollapseFactor ?? RUNG_ORDER.length)];
    }),
  );
  const controlUndercount = {
    declaredControlPairs: controlPairs,
    controlCells: controlCells.length,
    manifestDeclarations: (bodies.manifest.match(/"control":\s*\{/g) ?? []).length,
    citation:
      'scripts/generate-memory-matrix.mjs#overlaysFor pushes "control" only when ' +
      "`model[backend].control` is present",
    knownOmission: {
      pair: "krea_2_turbo|mlx",
      cellsItWouldAdd: cellsPerPairIfControlDeclared["krea_2_turbo|mlx"] ?? null,
      why:
        "SceneWorks ships BOTH control lanes — `crates/sceneworks-worker/src/image_jobs/" +
        "krea_control.rs` is documented in its own header as the MLX twin of " +
        "`generate_candle_krea_control_stream` in `krea_control_candle.rs` — but the manifest " +
        "declares `control` only in the candle block, so the MLX control cells do not exist in the " +
        "matrix at all. That is a manifest omission, not design intent.",
    },
    perPairCellCost: cellsPerPairIfControlDeclared,
    sensitivityNote:
      "Each additional (entry, backend) pair that declares `control` adds tiers x modes x rungs " +
      "cells, so the control count here is a FLOOR on demand rather than a measurement of it.",
  };

  // The published sweep domain, derived from the shipped plan rather than asserted.
  const planProviders = plan.providers.map((provider) => ({
    name: provider.name,
    backend: provider.backend,
    rung: provider.rung,
    evidenceScope: provider.evidenceScope,
    plannedCases: provider.cases.length,
    positiveCases: provider.cases.filter((item) => item.expectedResult === "passed").length,
  }));

  const model = {
    schemaVersion: 1,
    generatedFrom: {
      sceneWorksRevision: `source-tree:${sha256(sourceEntries.map(([name]) => bodies[name]).join(""))}`,
      matrixRevision: matrix.generatedFrom.sceneWorksRevision,
      inferenceRevision: matrix.generatedFrom.inferenceRevision,
      sources: Object.fromEntries(
        sourceEntries.map(([name, relative]) => [name, { path: relative, sha256: sha256(bodies[name]) }]),
      ),
    },
    confidenceTiers: {
      cells: "FACT — counted from docs/generated/memory-matrix.json",
      runs: "DERIVED — from the collapsing rule, every axis cited to code that was read",
      hours: "PARAMETERISED — per-run cost is an input with a swept range, not a measurement",
      producibility:
        "DERIVED — from the shipped plan, the evidence bundle, and the adapter sources; this is " +
        "the partition that says which of those hours can be spent today at all",
    },
    completedBaseline,
    producibility: producible,
    controlUndercount,
    cells: census,
    collapsing: {
      runUnit:
        "One `action:\"run\"` provider invocation. `runProviderPlan` spawns the provider command " +
        "once per planned case, so this is the unit that costs a model load.",
      runUnitCitation: "scripts/memory-calibration-harness.mjs#runProviderPlan",
      requiredScenariosPerRecord: REQUIRED_SCENARIOS.length,
      requiredScenarios: [...REQUIRED_SCENARIOS].sort(),
      axes: COLLAPSING_AXES,
      shippedPlan: planProviders,
    },
    runs,
    naSensitivity: {
      unknowable:
        "SC-15969 (the per-family rung-4 applicability survey) has not run, so the final " +
        "Structurally N/A count cannot be known. Only the `current` row below is a fact.",
      irreducibleFloor: {
        rung: "resident",
        cells: census.byRung.resident ?? 0,
        reason:
          "Rung 0 is the unoptimised baseline every model has by definition, so it can never be " +
          "Structurally N/A. No survey outcome can remove these cells from the campaign.",
      },
      scenarios: naSensitivity(cells),
    },
    wallClock: wallClock(runs, DEFAULT_PARAMETERS),
    fitVersusExhaustive: fitVersusExhaustive(cells, precedent, DEFAULT_PARAMETERS),
    parameters: DEFAULT_PARAMETERS,
    biggestUncertainties: [
      {
        input: "how much provider-adapter work the overlay axis needs",
        why:
          "It is the largest single bucket in the producibility partition and it gates 59.5% of the " +
          "matrix. Neither adapter can emit a truthful overlay scenario today, and the cost of " +
          "making them able to is code, not machine time — so it does not appear anywhere in the " +
          "wall-clock sweep. Until it is scoped, the schedule is unknowable regardless of how " +
          "precisely the run count is derived.",
        howToResolve:
          "Scope overlay support in both adapters, then re-run this model — the producibility " +
          "partition will move its `overlay-declined` cells to a later gate. Tracked as SC-16072; " +
          "the related control-declaration omission is SC-16073.",
        story: 16072,
      },
      {
        input: "perRunSeconds",
        why:
          "It is a pure multiplier on every reported hour and it spans a 40x range in the sweep. " +
          "The only measured anchor covers a VAE decode seam producing `gated` records, so it " +
          "does not bound a `complete` record from below in any useful way.",
        howToResolve:
          "Capture one `complete` record end to end on either backend and time it. One run " +
          "collapses this uncertainty entirely.",
      },
      {
        input: "whether the harness can amortise a model load across a target's rungs",
        why:
          "It is a clean 5x on the whole campaign and it is currently NOT realised: the runner " +
          "spawns a fresh process per planned case.",
        howToResolve:
          "Either batch a target's rungs into one provider invocation, or run the provider as a " +
          "warm daemon behind the argv command. Tracked as SC-16059.",
        story: 16059,
      },
      {
        input: "which geometry policy is true",
        why:
          "The binding logic accepts any resolution inside the envelope; the manifest says exactly " +
          "the opposite. The two policies differ by the geometry collapse factor across the " +
          "whole campaign.",
        howToResolve:
          "Decide it explicitly and make the losing side fail closed, rather than leaving two " +
          "contradictory policies in the same pipeline. Tracked as SC-16060.",
        story: 16060,
      },
    ],
  };

  return model;
}

function markdownTable(header, rows) {
  return [
    `| ${header.join(" | ")} |`,
    `| ${header.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${row.join(" | ")} |`),
  ];
}

function renderMarkdown(model) {
  const { cells, runs, fitVersusExhaustive: fit, producibility: prod, controlUndercount } = model;
  const lines = [
    "# Calibration cost model",
    "",
    "> Generated by `scripts/calibration-cost-model.mjs`. Do not edit by hand.",
    "",
    `- Matrix revision: \`${model.generatedFrom.matrixRevision}\``,
    `- Inference revision: \`${model.generatedFrom.inferenceRevision}\``,
    "",
    "**Cells are a fact. Runs are derived. Hours are parameterised.** Nothing below mixes the three.",
    "",
    "## 0. Read this first: the binding constraint is not GPU hours",
    "",
    `**${prod.headline}**`,
    "",
    `**${prod.behindProviderCode}** of ${cells.total} cells (**${round(prod.behindProviderCodeShare * 100, 1)}%**) are blocked by provider-adapter code that does not exist, not by machine time. **${prod.producibleToday}** could be certified by a run started now.`,
    "",
    ...markdownTable(
      ["Gate (first one that blocks the cell)", "Cells", "MLX", "Candle", "Blocked by"],
      prod.partition.map((gate) => [
        `\`${gate.id}\` — ${gate.label}`,
        String(gate.cells),
        String(gate.mlx),
        String(gate.candle),
        gate.blockedBy,
      ]),
    ),
    "",
    ...prod.partition
      .filter((gate) => gate.cells > 0)
      .map((gate) => `- \`${gate.id}\`: ${gate.citation}`),
    "",
    `There is also no completed-work baseline to subtract. ${model.completedBaseline.note}`,
    "",
    `The wall-clock sweep in section 5 is therefore an answer to "what would this cost once it is runnable", not "when will this be done".`,
    "",
    "## 1. Cells (fact)",
    "",
    ...markdownTable(
      ["Conformance state", "Cells"],
      Object.entries(cells.byState).map(([state, count]) => [`\`${state}\``, String(count)]),
    ),
    "",
    `Total **${cells.total}** cells across ${cells.sessions.total} (entry, backend, tier, mode, overlay) session keys, every one of which spans exactly ${cells.sessions.rungsPerSession.join("/")} rungs.`,
    "",
    ...markdownTable(
      ["Backend", "Cells", "Session keys"],
      Object.keys(cells.byBackend)
        .sort()
        .map((backend) => [
          backend,
          String(cells.byBackend[backend]),
          String(cells.sessions.byBackend[backend] ?? 0),
        ]),
    ),
    "",
    `Geometry is already collapsed: keyed per advertised resolution the matrix would hold **${cells.geometryExpandedCells}** cells, so the published count embeds a **${cells.geometryCollapseFactor}x** collapse.`,
    "",
    ...markdownTable(
      ["Overlay", "Cells"],
      Object.entries(cells.byOverlay).map(([overlay, count]) => [`\`${overlay}\``, String(count)]),
    ),
    "",
    `The overlay axis is **${cells.total - (cells.byOverlay.none ?? 0)}** cells — ${round(((cells.total - (cells.byOverlay.none ?? 0)) / cells.total) * 100, 1)}% of the matrix — and it is not uniformly real. \`lora\` and \`identity\` are genuine dimensions; \`control\` is nominal at ${controlUndercount.controlCells} cells on ${controlUndercount.declaredControlPairs.length} (entry, backend) pair(s).`,
    "",
    `**The control count is a floor, not demand.** ${controlUndercount.citation}, and the manifest declares it ${controlUndercount.manifestDeclarations} time(s). ${controlUndercount.knownOmission.why} Fixing that one omission adds **${controlUndercount.knownOmission.cellsItWouldAdd}** cells, doubling the control population. ${controlUndercount.sensitivityNote}`,
    "",
    "## 2. The collapsing rule (derived)",
    "",
    `The run unit is one \`action:"run"\` provider invocation — ${model.collapsing.runUnitCitation} spawns the provider command once per planned case, so that is what costs a model load. A \`complete\` record must pass all ${model.collapsing.requiredScenariosPerRecord} required scenarios plus quality and a measured negative mutation.`,
    "",
    ...markdownTable(
      ["Axis", "Verdict", "Factor"],
      model.collapsing.axes.map((axis) => [`\`${axis.axis}\``, axis.verdict, axis.factor]),
    ),
    "",
    ...model.collapsing.axes.flatMap((axis) => [
      `**\`${axis.axis}\` — ${axis.verdict}.** ${axis.rule}`,
      "",
      `> Citation: ${axis.citation}`,
      "",
      `> Caveat: ${axis.caveat}`,
      "",
    ]),
    "## 3. Runs (derived)",
    "",
    ...markdownTable(
      ["Unit", "Total", "MLX/Metal (this Mac)", "Candle/CUDA (rented)"],
      [
        [
          "Certifying records (1 per cell)",
          String(runs.certifyingRecords.total),
          String(runs.certifyingRecords.mlx),
          String(runs.certifyingRecords.candle),
        ],
        [
          `Warm sessions (model loads, ${runs.rungCollapseFactor} rungs each)`,
          String(runs.warmSessions.total),
          String(runs.warmSessions.mlx),
          String(runs.warmSessions.candle),
        ],
        [
          "Warm sessions with overlays also amortised into one load",
          String(runs.overlayAmortisedSessions.total),
          String(runs.overlayAmortisedSessions.mlx),
          String(runs.overlayAmortisedSessions.candle),
        ],
      ],
    ),
    "",
    `**Does the overlay axis evaporate?** On the record axis, no: each of the ${cells.total - (cells.byOverlay.none ?? 0)} non-\`none\` cells needs its own record, because \`overlay\` is in the cell key and \`validateComplete\` requires the overlay scenario to actually pass. On the model-load axis, mostly yes — dropping overlay from the session key takes loads from ${runs.warmSessions.total} to **${runs.overlayAmortisedSessions.total}**, a further **${runs.overlayLoadCollapseFactor}x**. That collapse is legitimate for a LoRA a warm generator can swap and **not** legitimate for an identity overlay that materialises extra encoder weights, so it is reported rather than claimed.`,
    "",
    `Excluded: ${runs.exemptions.alreadyStructurallyNotApplicable} cells already classified \`Structurally N/A\`, which the epic exempts from measurement. Nothing else is excluded — \`Missing\` cells still need a run, they just need an implementation first.`,
    "",
    `The warm-session row is **not currently achievable**. It is what the campaign would cost if one loaded generator covered a target's ${runs.rungCollapseFactor} rungs, at ${runs.recordsPerWarmSession} records per load; the harness spawns a fresh process per planned case today, so every rung pays its own model load.`,
    "",
    "## 4. Structurally N/A sensitivity",
    "",
    model.naSensitivity.unknowable,
    "",
    `Irreducible floor: the **${model.naSensitivity.irreducibleFloor.cells}** \`${model.naSensitivity.irreducibleFloor.rung}\` cells. ${model.naSensitivity.irreducibleFloor.reason}`,
    "",
    ...markdownTable(
      ["Scenario", "Kind", "Records", "MLX", "Candle", "Warm sessions"],
      model.naSensitivity.scenarios.map((scenario) => [
        `\`${scenario.name}\``,
        scenario.kind,
        String(scenario.certifyingRecords.total),
        String(scenario.certifyingRecords.mlx),
        String(scenario.certifyingRecords.candle),
        String(scenario.warmSessions.total),
      ]),
    ),
    "",
    `**The warm-session column does not move.** A session survives as long as any one of its rungs still needs measuring, and the resident baseline always does — so reclassifying rungs as \`Structurally N/A\` removes records but removes no model loads. If model load dominates the per-run cost, SC-15969's outcome changes the campaign's duration by far less than the record column suggests.`,
    "",
    "## 5. Wall clock (parameterised)",
    "",
    `Per-run seconds is an **input**, defaulting to ${model.wallClock.assumption.value} s. ${model.wallClock.assumption.status}. ${model.wallClock.assumption.reasoning}`,
    "",
    "### What was actually measured",
    "",
    ...model.wallClock.measuredAnchors.flatMap((anchor) => [
      `**${anchor.what}** — ${anchor.measurement}.`,
      "",
      `> How: ${anchor.derivation}`,
      "",
      `> Does NOT cover: ${anchor.doesNotCover}`,
      "",
    ]),
    "### Sweep",
    "",
    ...markdownTable(
      [
        "s/run",
        "Record-hours (total)",
        "MLX",
        "Candle",
        "Rungs amortised",
        "Rungs + overlays amortised",
      ],
      model.wallClock.sweep.map((row) => [
        String(row.perRunSeconds),
        String(row.recordHours.total),
        String(row.recordHours.mlx),
        String(row.recordHours.candle),
        String(row.warmSessionHours.total),
        String(row.overlayAmortisedSessionHours.total),
      ]),
    ),
    "",
    "Only the first three columns are achievable with the harness as written; the last two price collapses that do not exist yet. And per section 0, none of these hours can be spent today at all.",
    "",
    "## 6. Fit coefficients versus measure every cell",
    "",
    `The phase curve is **${fit.formula.kind}** — \`${fit.formula.expression}\` (${fit.formula.citation}), so ${fit.formula.coefficientsPerCurve} coefficients per curve and ${fit.formula.minimumPointsToDetermine} geometry points to determine one.`,
    "",
    `Worked precedent: **${fit.kreaPrecedent.measurementRecords}** measurement records produced ${fit.kreaPrecedent.curves} curves (${fit.kreaPrecedent.coefficients} coefficients) across tiers ${fit.kreaPrecedent.tiers.join(", ")}.`,
    "",
    `> ${fit.kreaPrecedent.honestyNote}`,
    "",
    ...markdownTable(
      ["Strategy", "Model loads", "Provider invocations (harness as written)"],
      fit.strategies.map((strategy) => [
        `\`${strategy.name}\``,
        String(strategy.sessions),
        String(strategy.providerInvocations),
      ]),
    ),
    "",
    ...fit.strategies.map((strategy) => `- \`${strategy.name}\`: ${strategy.policy}`),
    "",
    `Ratios: exact-geometry / fit = **${fit.ratios.exactGeometryOverFit}x**; exact-geometry / per-cell = ${fit.ratios.exactGeometryOverPerCell}x; fit / per-cell = ${fit.ratios.fitOverPerCell}x.`,
    "",
    `**${fit.finding}**`,
    "",
    "## 7. Biggest uncertainties",
    "",
    ...model.biggestUncertainties.flatMap((item) => [
      `**\`${item.input}\`** — ${item.why}`,
      "",
      `> Resolve by: ${item.howToResolve}`,
      "",
    ]),
  ];
  return `${lines.join("\n")}\n`;
}

async function main() {
  const model = await buildCostModel();
  const json = `${JSON.stringify(model, null, 2)}\n`;
  const markdown = renderMarkdown(model);
  if (process.argv.includes("--check")) {
    const [existingJson, existingMarkdown] = await Promise.all([
      readFile(path.join(ROOT, OUTPUT_JSON), "utf8"),
      readFile(path.join(ROOT, OUTPUT_MD), "utf8"),
    ]);
    if (canonicalSourceText(existingJson) !== json || canonicalSourceText(existingMarkdown) !== markdown) {
      throw new Error("generated calibration cost model is stale; run npm run generate:calibration-cost-model");
    }
    return;
  }
  await mkdir(path.join(ROOT, "docs/generated"), { recursive: true });
  await Promise.all([
    writeFile(path.join(ROOT, OUTPUT_JSON), json),
    writeFile(path.join(ROOT, OUTPUT_MD), markdown),
  ]);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
