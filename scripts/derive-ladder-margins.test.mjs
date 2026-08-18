import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  ACCOUNTING_DISCONTINUITY_THRESHOLD,
  BINDING_PHASE_ENVELOPE_SHARE,
  CANDLE_HARD_FLOOR,
  ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE,
  ESTIMATE_WIDENING_MULTIPLIER,
  MLX_HARD_FLOOR,
  RESIDUAL_BOUNDED_MAX_OVER_PHASES_EXEMPT_FROM_BINDING_PHASE_PIN,
  VARIANCE_SAFETY_MULTIPLIER,
  analyzeBackend,
  deriveBackendMargins,
  deriveMargins,
  loadEvidenceRecords,
} from "./derive-ladder-margins.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RUST_POLICY_PATH = path.join(ROOT, "crates", "sceneworks-worker", "src", "ladder_margin_policy.rs");
const SCRIPT_POLICY_PATH = path.join(ROOT, "scripts", "derive-ladder-margins.mjs");
const RUNBOOK_PATH = path.join(ROOT, "docs", "calibration-runbook.md");

/** The JSDoc block that ends immediately above `index` (whitespace-tolerant). */
function docCommentAbove(source, index) {
  const before = source.slice(0, index);
  const close = before.lastIndexOf("*/");
  assert.ok(close !== -1, "a doc comment closes above the declaration");
  assert.equal(before.slice(close + 2).trim(), "", "the doc comment sits immediately above");
  const open = before.lastIndexOf("/**", close);
  assert.ok(open !== -1, "the doc comment opens with /**");
  return before.slice(open, close + 2);
}

/** Extract `pub const NAME: f64 = VALUE;` pairs from the Rust policy module. */
async function rustConstants() {
  const source = await readFile(RUST_POLICY_PATH, "utf8");
  const constants = {};
  for (const match of source.matchAll(/pub const ([A-Z0-9_]+): f64 = ([0-9.]+);/g)) {
    constants[match[1]] = Number(match[2]);
  }
  return constants;
}

/** Minimal synthetic evidence record for exercising the derivation paths. */
function syntheticRecord({ id, backend = "mlx", tier = "q4", denoise, decode, overall }) {
  const phase = (bytes) => ({ activeBytes: bytes, allocatorBytes: bytes });
  return {
    id,
    backend,
    target: {
      modelId: "synthetic_model",
      provider: "synthetic_provider",
      mode: "text_to_image",
      tier,
      overlay: "none",
      geometry: { batch: 1, frames: 1, height: 1024, width: 1024 },
    },
    strategy: { rung: "bounded_decode", parameters: { decodeTileEdge: 512 } },
    observedMemory: {
      conditioning: phase(1_000),
      denoise: phase(denoise),
      decode: phase(decode),
      overall: phase(overall),
    },
    diagnostics: { measurements: [] },
  };
}

// The load-bearing pin: the constants landed in crates/sceneworks-worker must equal what the
// committed derivation computes from the committed evidence file. A perturbed Rust constant, a
// perturbed floor in the script, or evidence growth that pushes the variance term past a floor
// all land here as a red.
test("rust ladder_margin_policy constants match the derivation output", async () => {
  const derived = deriveMargins(await loadEvidenceRecords(ROOT));
  const constants = await rustConstants();

  assert.equal(constants.LADDER_MARGIN_HARD_FLOOR_MLX, derived.mlx.margins.hardFloor);
  assert.equal(constants.LADDER_MARGIN_HARD_FLOOR_CANDLE, derived.candle.margins.hardFloor);
  assert.equal(constants.MLX_STALE_MEASURED_MARGIN, derived.mlx.margins.staleMeasuredMargin);
  assert.equal(constants.MLX_ESTIMATE_MARGIN, derived.mlx.margins.estimateMargin);
  assert.equal(constants.CANDLE_STALE_MEASURED_MARGIN, derived.candle.margins.staleMeasuredMargin);
  assert.equal(constants.CANDLE_ESTIMATE_MARGIN, derived.candle.margins.estimateMargin);

  // Exactly the six derivation-coupled constants exist; a seventh margin constant added to the
  // Rust module without extending this pin would otherwise ship unpinned.
  assert.equal(Object.keys(constants).length, 6);
});

// Guards the honesty claims baked into the constants' doc comments. If evidence growth changes
// any of these (candle gains repeat pairs, intra-record repeat measurements appear), the doc
// comments and possibly the rule must be revisited rather than silently drifting.
test("derivation corpus facts backing the doc comments still hold", async () => {
  const derived = deriveMargins(await loadEvidenceRecords(ROOT));

  assert.ok(derived.mlx.analysis.repeatPairs > 0, "mlx repeat pairs exist");
  assert.equal(derived.mlx.margins.floorBinds, false, "mlx variance now exceeds the hard floor");
  assert.equal(derived.candle.analysis.repeatPairs, 0, "candle has zero repeat pairs (floor is the whole margin)");
  assert.equal(derived.mlx.analysis.intraRecordRepeatMeasurements, 0);
  assert.equal(derived.candle.analysis.intraRecordRepeatMeasurements, 0);

  // The 5% MLX floor's citable anchor (MLX_HARD_FLOOR rationale 2): the shipped predictor's
  // envelope gap really does span 4.76%..5.58% across all mlx records, and the floor never
  // exceeds the demonstrated headroom. Computed by the script, pinned here — not prose.
  const gap = derived.mlx.analysis.envelopeGap;
  assert.equal(gap.count, derived.mlx.analysis.recordCount, "every mlx record has a predicted overall peak");
  assert.equal((gap.min * 100).toFixed(2), "4.76", "envelope gap lower bound matches the doc comments");
  assert.equal((gap.max * 100).toFixed(2), "5.58", "envelope gap upper bound matches the doc comments");
  assert.ok(MLX_HARD_FLOOR <= gap.max, "mlx floor does not exceed the demonstrated envelope headroom");
});

// The issue-1 resolution (adversarial review of sc-18094): the non-binding exclusion is sound
// only for SAME-CELL admission, and the estimate margins do NOT absorb per-phase re-capture
// variance (a fully widened 68.55% term > the 50.4073% MLX estimate margin). That risk is carried
// by a pinned constraint instead: sc-18096/18097 must not admit estimate candidates whose
// predicted binding phase differs from the measured cell's without per-phase re-derivation.
// This test asserts the constraint constant exists on BOTH sides (script + Rust), that the Rust
// doc actually states the rule, and that the constraint is still load-bearing on the current
// corpus — if it ever stops being (spread drops under the margin), revisit whether the fold-in
// route has become viable.
test("the estimate-admission binding-phase constraint is pinned on both sides and load-bearing", async () => {
  assert.equal(ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE, true);

  const source = await readFile(RUST_POLICY_PATH, "utf8");
  const match = source.match(
    /pub const ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE: bool = (true|false);/,
  );
  assert.ok(match, "rust constraint constant exists");
  assert.equal(match[1] === "true", ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE);

  // The doc block immediately above the constant must state the rule, not just name it.
  const docBlock = source
    .slice(0, match.index)
    .split("\n")
    .reverse();
  const docLines = [];
  for (const line of docBlock) {
    if (line.trim() === "") continue;
    if (!line.trim().startsWith("///")) break;
    docLines.push(line);
  }
  const doc = docLines.join("\n");
  assert.ok(/MUST NOT/.test(doc), "doc states the prohibition");
  assert.ok(/binding phase/i.test(doc), "doc names the binding phase condition");
  assert.ok(/per-phase variance re-derivation/i.test(doc), "doc names the escape hatch");

  // The SCOPE sentence (re-review resolution): the constraint only governs candidates
  // extrapolated FROM a measured cell. The weights + headroom floor path of epic 18093 R1
  // (models with zero captures) has no measured binding phase to match and must stay
  // explicitly un-gated — pin the scoping sentence in BOTH docs so it cannot be dropped.
  const scriptSource = await readFile(SCRIPT_POLICY_PATH, "utf8");
  const scriptMatch = scriptSource.match(
    /export const ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE = (true|false);/,
  );
  assert.ok(scriptMatch, "script constraint constant exists");
  const scriptDoc = docCommentAbove(scriptSource, scriptMatch.index);
  for (const [name, text] of [["rust", doc], ["script", scriptDoc]]) {
    assert.ok(
      /NOT gated by this constraint/.test(text),
      `${name} doc scopes out no-measured-basis candidates`,
    );
    assert.ok(
      /weights \+ headroom floor/.test(text),
      `${name} doc names the epic 18093 R1 floor path that stays admissible`,
    );
  }

  // Load-bearing on the committed corpus: widening the demonstrated per-phase re-capture spread
  // by the derivation's safety and estimate factors exceeds the shipped MLX estimate margin, so
  // the constraint is still required for phase extrapolation.
  const derived = deriveMargins(await loadEvidenceRecords(ROOT));
  const canBind = derived.mlx.analysis.maxCanBindPhaseSpread;
  assert.equal((canBind.spread * 100).toFixed(4), "17.1369", "spread matches the number cited in both doc comments");
  const fullyWidenedCanBind = Math.max(MLX_HARD_FLOOR, canBind.spread * 2) * 2;
  assert.equal((fullyWidenedCanBind * 100).toFixed(2), "68.55", "fully widened spread matches the docs");
  assert.ok(fullyWidenedCanBind > derived.mlx.margins.estimateMargin, "constraint is load-bearing");
});

// SC-18829's fitted video curve is the narrow, ratified exception to the measured-binding-phase
// pin: every phase is independently fit and residual-bounded before the request-geometry max. Pin
// that exact construction on both sides, and pin the evidence verdict that it does not authorize a
// margin reduction. Otherwise a future scalar/provider-wide bypass could reuse the same boolean.
test("the residual-bounded max-over-phases exemption is structural and keeps margins unchanged", async () => {
  assert.equal(RESIDUAL_BOUNDED_MAX_OVER_PHASES_EXEMPT_FROM_BINDING_PHASE_PIN, true);

  const [rustSource, scriptSource, runbook, derived] = await Promise.all([
    readFile(RUST_POLICY_PATH, "utf8"),
    readFile(SCRIPT_POLICY_PATH, "utf8"),
    readFile(RUNBOOK_PATH, "utf8"),
    loadEvidenceRecords(ROOT).then(deriveMargins),
  ]);
  const rustMatch = rustSource.match(
    /pub const RESIDUAL_BOUNDED_MAX_OVER_PHASES_EXEMPT_FROM_BINDING_PHASE_PIN: bool = (true|false);/,
  );
  assert.ok(rustMatch, "rust exemption constant exists");
  assert.equal(
    rustMatch[1] === "true",
    RESIDUAL_BOUNDED_MAX_OVER_PHASES_EXEMPT_FROM_BINDING_PHASE_PIN,
  );
  const scriptMatch = scriptSource.match(
    /export const RESIDUAL_BOUNDED_MAX_OVER_PHASES_EXEMPT_FROM_BINDING_PHASE_PIN = (true|false);/,
  );
  assert.ok(scriptMatch, "script exemption constant exists");

  const rustDocLines = [];
  for (const line of rustSource.slice(0, rustMatch.index).split("\n").reverse()) {
    if (line.trim() === "") continue;
    if (!line.trim().startsWith("///")) break;
    rustDocLines.push(line);
  }
  const docs = [
    ["rust", rustDocLines.reverse().join("\n")],
    ["script", docCommentAbove(scriptSource, scriptMatch.index)],
  ];
  for (const [name, doc] of docs) {
    const prose = doc
      .replace(/^\s*(?:\/\/\/|\/\*\*?|\*\/|\*)\s?/gm, " ")
      .replace(/\s+/g, " ");
    assert.ok(/every phase independently/i.test(prose), `${name} doc requires independent phases`);
    assert.ok(/maximum fit\/held-out absolute residual/i.test(prose), `${name} doc requires residual bounds`);
    assert.ok(/maximum over phases/i.test(prose), `${name} doc requires max after residuals`);
    assert.ok(/ordinary backend estimate margin/i.test(prose), `${name} doc retains the normal margin`);
    assert.ok(/not provider-wide|not a provider-wide/i.test(prose), `${name} doc rejects a provider-wide bypass`);
  }

  // SC-18829's claim is NON-REDUCTION: the video ratification may not shrink the image-lane
  // margins. The exact MLX value is owned by the sc-18094 derivation pin above (0.5040734… on the
  // merged 89-record corpus) — re-pinning it here would just freeze the corpus twice.
  assert.ok(
    derived.mlx.margins.estimateMargin >= 0.10,
    "ratification does not reduce MLX margin",
  );
  assert.equal(derived.candle.margins.estimateMargin, 0.04, "ratification does not reduce candle margin");
  assert.match(runbook, /Admission-margin verdict \(SC-18829\): keep the ratified constants unchanged/);
  assert.match(runbook, /largest\s+adopted q8 `cross` residual is 0\.4438 GiB/);
  assert.match(runbook, /10% MLX estimate margin/);
  assert.match(runbook, /Candle has no\s+promoted temporal curve yet/);
});

// Mutation-proofs the VARIANCE path: with a synthetic repeat pair whose doubled spread exceeds
// the floor, the derived margin must track the variance term, not the floor. Without this, a
// derivation that always returned the floor would still pass the pin above.
test("variance term overrides the floor when repeat spread is wide enough", () => {
  const spread = 0.06; // x2 => 12%, above the 5% mlx floor
  const analysis = analyzeBackend([
    syntheticRecord({ id: "syn-a", denoise: 10_000_000_000, decode: 9_000_000_000, overall: 10_500_000_000 }),
    syntheticRecord({
      id: "syn-b",
      denoise: Math.round(10_000_000_000 * (1 + spread)),
      decode: 9_000_000_000,
      overall: 10_500_000_000,
    }),
  ]);
  const margins = deriveBackendMargins(analysis, MLX_HARD_FLOOR);

  assert.equal(analysis.repeatPairs, 1);
  assert.ok(!margins.floorBinds);
  assert.ok(Math.abs(margins.staleMeasuredMargin - VARIANCE_SAFETY_MULTIPLIER * spread) < 1e-9);
  assert.ok(
    Math.abs(margins.estimateMargin - ESTIMATE_WIDENING_MULTIPLIER * margins.staleMeasuredMargin) < 1e-9,
  );
});

test("a backend with no repeat pairs falls back to the hard floor as the whole margin", () => {
  const analysis = analyzeBackend([
    syntheticRecord({ id: "syn-a", backend: "candle", denoise: 4e9, decode: 4e9, overall: 4e9 }),
    syntheticRecord({ id: "syn-b", backend: "candle", tier: "q8", denoise: 8e9, decode: 8e9, overall: 8e9 }),
  ]);
  const margins = deriveBackendMargins(analysis, CANDLE_HARD_FLOOR);

  assert.equal(analysis.repeatPairs, 0);
  assert.equal(margins.maxBindingSpread, null);
  assert.ok(margins.floorBinds);
  assert.equal(margins.staleMeasuredMargin, CANDLE_HARD_FLOOR);
  assert.equal(margins.estimateMargin, ESTIMATE_WIDENING_MULTIPLIER * CANDLE_HARD_FLOOR);
});

// The two exclusion rules that keep the corpus's known artifacts out of the SAME-CELL variance
// estimate: harness accounting flips (~44 KB vs ~16 GB conditioning) and spreads on phases too
// far below the envelope to bind a same-cell admission (the 17% rung-4 denoise swing on a 2 GB
// phase under a 16 GB envelope). The non-binding exclusion is scoped: the excluded swing must
// still surface as maxCanBindPhaseSpread, because under estimate-backed extrapolation that
// phase CAN bind — that is what the binding-phase constraint above exists for.
test("accounting flips and non-binding phase spreads are excluded from the variance estimate", () => {
  const belowBinding = 0.4 * 10_500_000_000; // < BINDING_PHASE_ENVELOPE_SHARE of the envelope
  const analysis = analyzeBackend([
    syntheticRecord({ id: "syn-a", denoise: belowBinding, decode: 44_000, overall: 10_500_000_000 }),
    syntheticRecord({
      id: "syn-b",
      denoise: Math.round(belowBinding * 1.2),
      decode: 16_000_000_000, // > ACCOUNTING_DISCONTINUITY_THRESHOLD apart from 44 KB
      overall: 10_500_000_000,
    }),
  ]);

  assert.ok(BINDING_PHASE_ENVELOPE_SHARE > 0.4);
  assert.ok(16_000_000_000 / 44_000 - 1 > ACCOUNTING_DISCONTINUITY_THRESHOLD);
  assert.ok(analysis.accountingFlipsExcluded >= 2, "decode flip excluded on both metrics");
  assert.ok(analysis.maxNonBindingSpread.spread > 0.19, "denoise swing recorded as non-binding");
  assert.ok(
    analysis.bindingSpreads.every((entry) => entry.phase === "overall" || entry.phase === "conditioning"),
    "neither excluded spread reached the binding set",
  );
  assert.equal(analysis.maxCanBindPhaseSpread.phase, "denoise");
  assert.ok(
    Math.abs(analysis.maxCanBindPhaseSpread.spread - 0.2) < 0.01,
    "the same-cell-excluded denoise swing still surfaces as the max can-bind phase spread",
  );
});
