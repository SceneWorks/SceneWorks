import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  ACCOUNTING_DISCONTINUITY_THRESHOLD,
  BINDING_PHASE_ENVELOPE_SHARE,
  CANDLE_HARD_FLOOR,
  ESTIMATE_WIDENING_MULTIPLIER,
  MLX_HARD_FLOOR,
  VARIANCE_SAFETY_MULTIPLIER,
  analyzeBackend,
  deriveBackendMargins,
  deriveMargins,
  loadEvidenceRecords,
} from "./derive-ladder-margins.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RUST_POLICY_PATH = path.join(ROOT, "crates", "sceneworks-worker", "src", "ladder_margin_policy.rs");

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
  assert.ok(derived.mlx.margins.floorBinds, "mlx floor binds (doc comment says variance term < floor)");
  assert.equal(derived.candle.analysis.repeatPairs, 0, "candle has zero repeat pairs (floor is the whole margin)");
  assert.equal(derived.mlx.analysis.intraRecordRepeatMeasurements, 0);
  assert.equal(derived.candle.analysis.intraRecordRepeatMeasurements, 0);
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

// The two exclusion rules that keep the corpus's known artifacts out of the variance estimate:
// harness accounting flips (~44 KB vs ~16 GB conditioning) and spreads on phases too far below
// the envelope to ever bind admission (the 17% rung-4 denoise swing on a 2 GB phase under a
// 16 GB envelope).
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
});
