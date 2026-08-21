#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const SC19057_INFERENCE_REVISION = "4013049764172ee7dc707101c7da8c83c1483f2d";
export const SC19057_INFERENCE_CLOSURE =
  "7f6a6864040718ec01c9de41db34fc627c22265c6b79babf6c6e7490db2bf520";
export const SC19057_WAN_REPOSITORY = "SceneWorks/wan2.2-ti2v-5b-candle";
export const SC19057_WAN_REVISION = "9b173dc8660334a87a11e67de58939afe68f8cb2";
export const SC19057_CALIBRATION_FINGERPRINT =
  "sc-19223-wan2-2-ti2v-5b-candle-sequential-load-v1";

export const SC19057_CASES = Object.freeze([
  Object.freeze({ fixture: "wan2-2-ti2v-5b-candle-q4-832x480-f93-fps24-seed19057", role: "fit", width: 832, height: 480, frames: 93 }),
  Object.freeze({ fixture: "wan2-2-ti2v-5b-candle-q4-832x480-f141-fps24-seed19057", role: "fit", width: 832, height: 480, frames: 141 }),
  Object.freeze({ fixture: "wan2-2-ti2v-5b-candle-q4-832x480-f189-fps24-seed19057", role: "held_out", width: 832, height: 480, frames: 189 }),
  Object.freeze({ fixture: "wan2-2-ti2v-5b-candle-q4-1280x704-f93-fps24-seed19057", role: "fit", width: 1280, height: 704, frames: 93 }),
  Object.freeze({ fixture: "wan2-2-ti2v-5b-candle-q4-1280x704-f141-fps24-seed19057", role: "held_out", width: 1280, height: 704, frames: 141 }),
  Object.freeze({ fixture: "wan2-2-ti2v-5b-candle-q4-704x1280-f189-fps24-seed19057", role: "fit", width: 704, height: 1280, frames: 189 }),
]);

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_PLAN = path.join(
  ROOT,
  "docs/calibration/sc-19057/wan-candle-video-capture-plan.json",
);
const EXACT_ENGAGED_RUNGS = ["resident", "staged_residency"];

function fail(message) {
  throw new Error(`SC-19057 Wan capture acceptance failed: ${message}`);
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) fail(`${label} must be ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

function assertExactArray(actual, expected, label) {
  if (!Array.isArray(actual) || JSON.stringify(actual) !== JSON.stringify(expected)) {
    fail(`${label} must be exactly ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function assertExactCaseShape(value, expected, prefix, roleField) {
  assertEqual(value.fixture, expected.fixture, `${prefix}.fixture`);
  if (roleField) assertEqual(value[roleField], expected.role, `${prefix}.${roleField}`);
  assertEqual(value.backend, "candle", `${prefix}.backend`);
  assertEqual(value.evidenceScope, "authoritative", `${prefix}.evidenceScope`);
  assertEqual(value.loadShape, "eager_materialization", `${prefix}.loadShape`);
  assertEqual(value.target?.provider, "wan2_2_ti2v_5b", `${prefix}.target.provider`);
  assertEqual(value.target?.modelId, "wan_2_2", `${prefix}.target.modelId`);
  assertEqual(value.target?.tier, "q4", `${prefix}.target.tier`);
  assertEqual(value.target?.mode, "text_to_video", `${prefix}.target.mode`);
  assertEqual(value.target?.overlay, "none", `${prefix}.target.overlay`);
  assertEqual(value.target?.geometry?.width, expected.width, `${prefix}.target.geometry.width`);
  assertEqual(value.target?.geometry?.height, expected.height, `${prefix}.target.geometry.height`);
  assertEqual(value.target?.geometry?.frames, expected.frames, `${prefix}.target.geometry.frames`);
  assertEqual(value.target?.geometry?.batch, 1, `${prefix}.target.geometry.batch`);
  assertEqual(value.calibrationFingerprint, SC19057_CALIBRATION_FINGERPRINT, `${prefix}.calibrationFingerprint`);
}

export function assertSc19057Plan(plan) {
  if (!Array.isArray(plan?.providers)) fail("the plan must carry a providers array");
  assertEqual(plan.providers.length, SC19057_CASES.length, "plan.providers.length");
  const fixtureSet = new Set(plan.providers.map(({ fixture }) => fixture));
  assertEqual(fixtureSet.size, SC19057_CASES.length, "unique plan fixture count");

  SC19057_CASES.forEach((expected, index) => {
    const provider = plan.providers[index];
    assertExactCaseShape(provider, expected, `plan.providers[${index}]`, "_role");
    assertEqual(provider.name, `candle-wan-2-2-ti2v-5b-q4-${expected.width}x${expected.height}-f${expected.frames}`, `plan.providers[${index}].name`);
    assertEqual(provider.rung, "staged_residency", `plan.providers[${index}].rung`);
    assertExactArray(provider.engagedRungs, EXACT_ENGAGED_RUNGS, `plan.providers[${index}].engagedRungs`);
    if (
      !Array.isArray(provider.cases) ||
      provider.cases.length !== 1 ||
      Object.keys(provider.cases[0]?.parameters ?? {}).length !== 0 ||
      provider.cases[0]?.expectedResult !== "passed"
    ) {
      fail(`plan.providers[${index}].cases must be the single empty-parameter passed case`);
    }
  });
}

export function validateSc19057WanCapture({
  bundle,
  plan,
  expectedSceneWorksRevision,
  expectedInferenceRevision = SC19057_INFERENCE_REVISION,
}) {
  assertSc19057Plan(plan);
  if (!Array.isArray(bundle?.records)) fail("the capture bundle must carry a records array");
  assertEqual(bundle.records.length, SC19057_CASES.length, "record count");

  const fixtureSet = new Set(bundle.records.map(({ fixture }) => fixture));
  const recordIdSet = new Set(bundle.records.map(({ id }) => id));
  assertEqual(fixtureSet.size, SC19057_CASES.length, "unique captured fixture count");
  assertEqual(recordIdSet.size, SC19057_CASES.length, "unique record id count");
  if ([...recordIdSet].some((id) => typeof id !== "string" || id.length === 0)) {
    fail("every record id must be a non-empty string");
  }

  const expectedScene = expectedSceneWorksRevision ?? bundle.records[0]?.repositories?.sceneWorks?.revision;
  if (!/^[0-9a-f]{40}$/.test(expectedScene ?? "")) {
    fail(`expected SceneWorks revision must resolve to an exact lowercase 40-hex commit, got ${JSON.stringify(expectedScene)}`);
  }
  assertEqual(expectedInferenceRevision, SC19057_INFERENCE_REVISION, "expected inference revision");

  SC19057_CASES.forEach((expected, index) => {
    const record = bundle.records.find(({ fixture }) => fixture === expected.fixture);
    if (!record) fail(`missing required fixture ${expected.fixture}`);
    assertExactCaseShape(record, expected, `records[${index}]`, null);
    assertEqual(record.status, "runtime_complete", `records[${index}].status`);
    assertEqual(record.strategy?.rung, "staged_residency", `records[${index}].strategy.rung`);
    assertExactArray(record.strategy?.engagedRungs, EXACT_ENGAGED_RUNGS, `records[${index}].strategy.engagedRungs`);
    assertEqual(record.artifact?.repository, SC19057_WAN_REPOSITORY, `records[${index}].artifact.repository`);
    assertEqual(record.artifact?.resolvedRevision, SC19057_WAN_REVISION, `records[${index}].artifact.resolvedRevision`);
    assertEqual(record.artifact?.variant, "q4", `records[${index}].artifact.variant`);
    assertEqual(record.repositories?.sceneWorks?.dirty, false, `records[${index}].repositories.sceneWorks.dirty`);
    assertEqual(record.repositories?.sceneWorks?.revision, expectedScene, `records[${index}].repositories.sceneWorks.revision`);
    assertEqual(record.repositories?.inference?.dirty, false, `records[${index}].repositories.inference.dirty`);
    assertEqual(record.repositories?.inference?.revision, expectedInferenceRevision, `records[${index}].repositories.inference.revision`);
    assertEqual(record.repositories?.inference?.closureDigest, SC19057_INFERENCE_CLOSURE, `records[${index}].repositories.inference.closureDigest`);
  });

  return {
    story: "SC-19057",
    lane: "candle:wan2_2_ti2v_5b",
    plannedEntries: SC19057_CASES.length,
    capturedFixtures: fixtureSet.size,
    runtimeComplete: bundle.records.filter(({ status }) => status === "runtime_complete").length,
    roles: { fit: 4, heldOut: 2 },
    sceneWorksRevision: expectedScene,
    inferenceRevision: expectedInferenceRevision,
    inferenceClosureDigest: SC19057_INFERENCE_CLOSURE,
    artifact: {
      repository: SC19057_WAN_REPOSITORY,
      revision: SC19057_WAN_REVISION,
      variant: "q4",
    },
  };
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function value(args, flag, fallback) {
  const index = args.indexOf(flag);
  if (index < 0) return fallback;
  if (!args[index + 1] || args[index + 1].startsWith("--")) fail(`${flag} requires a value`);
  return args[index + 1];
}

async function main() {
  const args = process.argv.slice(2);
  const allowed = new Set([
    "--input",
    "--plan",
    "--sceneworks-revision",
    "--inference-revision",
    "--write-receipt",
  ]);
  for (let index = 0; index < args.length; index += 2) {
    if (!allowed.has(args[index])) fail(`unknown argument ${JSON.stringify(args[index])}`);
  }
  const inputPath = value(args, "--input");
  if (!inputPath) fail("--input is required");
  const planPath = value(args, "--plan", DEFAULT_PLAN);
  const expectedSceneWorksRevision = value(args, "--sceneworks-revision");
  const expectedInferenceRevision = value(
    args,
    "--inference-revision",
    SC19057_INFERENCE_REVISION,
  );
  const [bundleRaw, planRaw] = await Promise.all([
    readFile(path.resolve(inputPath)),
    readFile(path.resolve(planPath)),
  ]);
  const receipt = validateSc19057WanCapture({
    bundle: JSON.parse(bundleRaw),
    plan: JSON.parse(planRaw),
    expectedSceneWorksRevision,
    expectedInferenceRevision,
  });
  receipt.captureSha256 = sha256(bundleRaw);
  receipt.planSha256 = sha256(planRaw);
  const receiptPath = value(args, "--write-receipt");
  if (receiptPath) await writeFile(path.resolve(receiptPath), `${JSON.stringify(receipt, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error.stack ?? error.message);
    process.exitCode = 1;
  });
}
