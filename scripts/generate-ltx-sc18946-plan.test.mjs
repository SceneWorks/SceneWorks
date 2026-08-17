import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildPlans,
  FINGERPRINT,
  INFERENCE_REVISION,
  INCIDENT_PREDICTED_DECODE_BYTES,
  Q4_F305_CRASH_FOOTPRINT_BYTES,
  RATIFIED_BOUNDED_FIXTURE,
  RATIFIED_BOUNDED_PARAMETERS,
  RATIFIED_BOUNDED_PROVIDER,
  RATIFIED_BOUNDED_TIERS,
  RUNG2_PARAMETERS,
  SAFETY_DISPOSITIONS,
  SEED,
  TIER_INVENTORY_BYTES,
  TIERS,
} from "./generate-ltx-sc18946-plan.mjs";

const plans = buildPlans();
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

test("the committed SC-18946 plans are derived byte-for-byte", async () => {
  for (const [name, plan] of Object.entries(plans)) {
    const actual = await readFile(path.join(ROOT, "docs/calibration/sc-18946", name), "utf8");
    assert.equal(actual, `${JSON.stringify(plan, null, 2)}\n`, name);
  }
});

test("the single-pass campaign has an identical fit and held-out matrix in all tiers", () => {
  const rows = plans["ltx-mlx-single-pass-sweep.json"].providers;
  assert.equal(rows.length, 54);
  const signature = (row) => [
    row.target.geometry.width,
    row.target.geometry.height,
    row.target.geometry.frames,
    row.fixture.match(/-fps(\d+)-/)[1],
    row._role,
    row._campaignPhase,
  ].join(":");
  const byTier = Object.fromEntries(TIERS.map((tier) => [
    tier,
    rows.filter((row) => row.target.tier === tier).map(signature),
  ]));
  assert.deepEqual(byTier.q4, byTier.q8);
  assert.deepEqual(byTier.q8, byTier.bf16);
  for (const tier of TIERS) {
    const tierRows = rows.filter((row) => row.target.tier === tier);
    assert.equal(tierRows.filter((row) => row._role === "fit").length, 14, `${tier} fit rows`);
    assert.equal(tierRows.filter((row) => row._role === "held_out").length, 3, `${tier} held-out rows`);
    assert.equal(tierRows.filter((row) => row._role === "held_out_fps_probe").length, 1, `${tier} fps probe`);
  }
  assert.equal(rows.filter((row) => row._role.startsWith("held_out")).length, 12);
  assert.ok(rows.every((row) => row.rung === "staged_residency"));
  assert.ok(rows.every((row) => row.cases[0].parameters && Object.keys(row.cases[0].parameters).length === 0));
});

test("every campaign identity and derived geometry fact is exact and collision-free", () => {
  const rows = Object.values(plans).flatMap((plan) => plan.providers);
  assert.equal(new Set(rows.map((row) => row.name)).size, rows.length, "driver names are globally unique");
  assert.equal(new Set(rows.map((row) => row.fixture)).size, rows.length, "fixtures are globally unique");
  for (const plan of Object.values(plans)) {
    assert.deepEqual(plan.campaignIdentity, {
      inferenceRevision: INFERENCE_REVISION,
      calibrationFingerprint: FINGERPRINT,
      seed: SEED,
    });
  }
  for (const row of rows) {
    const { width, height, frames } = row.target.geometry;
    assert.equal((frames - 1) % 8, 0, `${row.name} is on the causal 1 + 8k frame lattice`);
    assert.equal(width % 64, 0, `${row.name} width is within the provider envelope`);
    assert.equal(height % 64, 0, `${row.name} height is within the provider envelope`);
    assert.equal(row._outputVoxels, width * height * frames, `${row.name} output voxels`);
    assert.equal(
      row._latentTokens,
      ((frames - 1) / 8 + 1) * (width / 32) * (height / 32),
      `${row.name} latent tokens`,
    );
    assert.equal(row._writableFrameCap, Math.floor(2_147_483_647 / (8 * width * height)));
    assert.match(row.fixture, new RegExp(`-${row.target.tier}-${width}x${height}-f${frames}-fps\\d+(?:-bounded-decode-192-64)?-seed${SEED}$`));
    assert.equal(row.calibrationFingerprint, FINGERPRINT);
    assert.equal(row.cases.length, 1);
    assert.equal(row.cases[0].expectedResult, "passed");
  }
});

test("the flip bands, deep anchors, original six and held-out transfer cells are frozen", () => {
  const q8 = plans["ltx-mlx-single-pass-sweep.json"].providers
    .filter((row) => row.target.tier === "q8");
  const cells = new Set(q8.map((row) => `${row.target.geometry.width}x${row.target.geometry.height}:f${row.target.geometry.frames}:${row._campaignPhase}`));
  for (const cell of [
    "768x512:f121:original_fit", "768x512:f241:original_fit", "768x512:f361:original_fit",
    "1280x704:f121:original_fit", "1280x704:f145:original_fit", "1280x704:f177:original_fit",
    "768x512:f97:deep_conditioning", "640x640:f97:deep_conditioning", "1280x704:f97:deep_conditioning",
    "768x512:f257:flip_band", "768x512:f265:flip_band",
    "640x640:f249:flip_band", "640x640:f257:flip_band", "1280x704:f113:flip_band",
    "640x640:f177:coefficient_transfer", "512x768:f361:coefficient_transfer", "704x1280:f177:coefficient_transfer",
  ]) assert.ok(cells.has(cell), cell);
});

test("rung two preserves the six 384 by 64 rows and appends three matched bounded anchors", () => {
  const rows = plans["ltx-mlx-rung2-sweep.json"].providers;
  assert.equal(rows.length, 9);
  const originalSeventyOne = Object.entries(plans).flatMap(([name, plan]) =>
    name === "ltx-mlx-rung2-sweep.json" ? plan.providers.slice(0, 7) : plan.providers);
  assert.equal(originalSeventyOne.length, 71);
  assert.equal(
    createHash("sha256").update(JSON.stringify(originalSeventyOne)).digest("hex"),
    "473fc389fc12a2ffd4ae2eb1680f24033fc48a5c55503951570124d69b3d97f6",
    "all 71 pre-SC-20430 rows must remain byte-identical and ordered",
  );
  assert.equal(
    createHash("sha256").update(JSON.stringify(rows.slice(0, 7))).digest("hex"),
    "1618f6cb8a0589652da550a3cbd27f472c36748995a141cc9199d7847ebdf64a",
    "the pre-SC-20430 rung-2 rows must remain byte-identical and ordered",
  );
  const frozen = rows.slice(0, 6);
  const ratified = rows.slice(6);
  assert.deepEqual(TIERS, ["q4", "q8", "bf16"]);
  assert.deepEqual(
    frozen.map((row) => `${row.target.geometry.frames}:${row.target.tier}`),
    ["305:q4", "305:q8", "305:bf16", "449:q4", "449:q8", "449:bf16"],
    "the checked-in matrix preserves the reviewed serialized launch order",
  );
  assert.deepEqual(new Set(frozen.map((row) => row.target.tier)), new Set(TIERS));
  assert.deepEqual(new Set(frozen.map((row) => row.target.geometry.frames)), new Set([305, 449]));
  for (const row of frozen) {
    assert.equal(row.rung, "bounded_decode");
    assert.deepEqual(row.engagedRungs, ["resident", "staged_residency", "bounded_decode"]);
    assert.deepEqual(row.cases, [{ parameters: RUNG2_PARAMETERS, expectedResult: "passed" }]);
    assert.equal(row.calibrationFingerprint, FINGERPRINT);
    assert.match(row.fixture, new RegExp(`seed${SEED}$`));
  }
  const predicted = Object.fromEntries(frozen.filter((row) => row.target.tier === "q8").map((row) => [row.target.geometry.frames, row._predictedDecodeBytes]));
  assert.equal(predicted[305], 18_540_396_800);
  assert.equal(predicted[449], 23_730_848_000);
  assert.deepEqual(ratified.map((row) => row.target.tier), RATIFIED_BOUNDED_TIERS);
  assert.equal(ratified[0].name, RATIFIED_BOUNDED_PROVIDER);
  assert.equal(ratified[0].fixture, RATIFIED_BOUNDED_FIXTURE);
  for (const row of ratified) {
    assert.equal(row._role, "bounded_carrier_entry");
    assert.equal(row._campaignPhase, "bounded_carrier_ratification");
    assert.deepEqual(row.target.geometry, {
      width: 768, height: 512, batch: 1, frames: 121,
    });
    assert.deepEqual(row.cases, [{
      parameters: RATIFIED_BOUNDED_PARAMETERS, expectedResult: "passed",
    }]);
    assert.equal(row._measurementSafety.disposition, SAFETY_DISPOSITIONS.SAFETY_REFUSED_OPEN);
    assert.equal(row._measurementSafety.predictedDecodeBytes, 6_264_848_640);
    assert.equal(
      row._measurementSafety.incidentCalibratedProjectionBytes,
      84_694_536_320 + TIER_INVENTORY_BYTES[row.target.tier] - TIER_INVENTORY_BYTES.q4,
    );
  }
});

test("host-risk coverage is explicitly planned rather than absent", () => {
  const rows = plans["ltx-mlx-host-risk-probes.json"].providers;
  assert.equal(rows.length, 10);
  assert.ok(rows.every((row) => row._coverageExpectation === "captured_or_attempted_or_arithmetic_unmeasurable"));
  for (const tier of TIERS) {
    const signature = new Set(rows
      .filter((row) => row.target.tier === tier && row._role === "planned_host_risk")
      .map((row) => `${row.target.geometry.width}x${row.target.geometry.height}:f${row.target.geometry.frames}`));
    assert.deepEqual(signature, new Set(["768x512:f449", "1280x704:f241", "1280x704:f297"]), tier);
  }
  const reproductions = rows.filter((row) => row._role === "reproduction_probe");
  assert.equal(reproductions.length, 1);
  assert.equal(reproductions[0].target.tier, "bf16");
  assert.deepEqual(reproductions[0].target.geometry, { width: 768, height: 512, batch: 1, frames: 145 });
  assert.match(reproductions[0].fixture, /-fps24-seed18946$/);
});

test("SC-19642 classifies every row for pre-load refusal without inventing a safe host fraction", () => {
  const rows = Object.values(plans).flatMap((plan) => plan.providers);
  assert.equal(rows.length, 73);
  assert.equal(Q4_F305_CRASH_FOOTPRINT_BYTES, 96_970_084_480);
  assert.deepEqual(TIER_INVENTORY_BYTES, {
    q4: 20_467_690_460,
    q8: 29_728_720_716,
    bf16: 47_092_811_992,
  });
  for (const row of rows) {
    const safety = row._measurementSafety;
    assert.ok(Object.values(SAFETY_DISPOSITIONS).includes(safety.disposition));
    assert.equal(safety.tierInventoryBytes, TIER_INVENTORY_BYTES[row.target.tier]);
    assert.equal(safety.incidentCrashFootprintBytes, Q4_F305_CRASH_FOOTPRINT_BYTES);
    assert.equal(safety.incidentPredictedDecodeBytes, INCIDENT_PREDICTED_DECODE_BYTES);
    assert.equal(
      safety.incidentCalibratedProjectionBytes,
      Q4_F305_CRASH_FOOTPRINT_BYTES
        + (TIER_INVENTORY_BYTES[row.target.tier] - TIER_INVENTORY_BYTES.q4)
        + (safety.predictedDecodeBytes - INCIDENT_PREDICTED_DECODE_BYTES),
    );
    assert.ok(safety.projectionAssumptions.some((claim) => claim.includes("not a physical-footprint bound")));
  }

  // Mutation locks the safety terminal itself, not merely the presence of descriptive metadata.
  const mutated = structuredClone(rows[0]);
  mutated._measurementSafety.disposition = "capturable";
  assert.ok(!Object.values(SAFETY_DISPOSITIONS).includes(mutated._measurementSafety.disposition));

  const rung2 = plans["ltx-mlx-rung2-sweep.json"].providers;
  assert.equal(rung2[0]._measurementSafety.disposition, SAFETY_DISPOSITIONS.INCIDENT_FORBIDDEN);
  assert.equal(rung2[3]._measurementSafety.disposition, SAFETY_DISPOSITIONS.ARITHMETIC_UNMEASURABLE);
  assert.ok(rung2.filter((_, index) => index !== 0 && index !== 3).every((row) =>
    row._measurementSafety.disposition === SAFETY_DISPOSITIONS.SAFETY_REFUSED_OPEN));
  assert.ok(rows.filter((row) => row.rung === "staged_residency")
    .every((row) => row._measurementSafety.disposition === SAFETY_DISPOSITIONS.SAFETY_REFUSED_OPEN));
});
