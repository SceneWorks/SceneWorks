import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  SC19057_CALIBRATION_FINGERPRINT,
  SC19057_CASES,
  SC19057_INFERENCE_CLOSURE,
  SC19057_INFERENCE_REVISION,
  SC19057_WAN_REPOSITORY,
  SC19057_WAN_REVISION,
  validateSc19057WanCapture,
} from "./validate-sc19057-wan-capture.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PLAN_PATH = path.join(
  ROOT,
  "docs/calibration/sc-19057/wan-candle-video-capture-plan.json",
);
const PLAN = JSON.parse(readFileSync(PLAN_PATH, "utf8"));
const SCENEWORKS_REVISION = "81be365fc9e9c2fb609b0c5c38be9fe4442991a3";

function recordOf(entry, index) {
  return {
    id: `imc-sc19057-${String(index).padStart(2, "0")}`,
    fixture: entry.fixture,
    status: "runtime_complete",
    backend: "candle",
    evidenceScope: "authoritative",
    loadShape: "eager_materialization",
    calibrationFingerprint: SC19057_CALIBRATION_FINGERPRINT,
    target: {
      provider: "wan2_2_ti2v_5b",
      modelId: "wan_2_2",
      tier: "q4",
      mode: "text_to_video",
      overlay: "none",
      geometry: {
        width: entry.width,
        height: entry.height,
        frames: entry.frames,
        batch: 1,
      },
    },
    strategy: {
      rung: "staged_residency",
      engagedRungs: ["resident", "staged_residency"],
    },
    artifact: {
      repository: SC19057_WAN_REPOSITORY,
      resolvedRevision: SC19057_WAN_REVISION,
      variant: "q4",
    },
    repositories: {
      sceneWorks: { revision: SCENEWORKS_REVISION, dirty: false },
      inference: {
        revision: SC19057_INFERENCE_REVISION,
        dirty: false,
        closureDigest: SC19057_INFERENCE_CLOSURE,
      },
    },
  };
}

function validBundle() {
  return { records: SC19057_CASES.map(recordOf) };
}

test("SC-19057 accepts only the exact six authoritative runtime-complete Wan q4 cases", () => {
  const receipt = validateSc19057WanCapture({
    bundle: validBundle(),
    plan: PLAN,
    expectedSceneWorksRevision: SCENEWORKS_REVISION,
  });
  assert.deepEqual(
    {
      plannedEntries: receipt.plannedEntries,
      capturedFixtures: receipt.capturedFixtures,
      runtimeComplete: receipt.runtimeComplete,
      roles: receipt.roles,
    },
    { plannedEntries: 6, capturedFixtures: 6, runtimeComplete: 6, roles: { fit: 4, heldOut: 2 } },
  );
});

test("SC-19057 capture validation kills count, uniqueness, geometry, identity and authority mutations", () => {
  const mutations = [
    ["missing sixth row", (bundle) => { bundle.records.pop(); }, /record count/],
    ["duplicate fixture", (bundle) => { bundle.records[5].fixture = bundle.records[0].fixture; }, /unique captured fixture count/],
    ["duplicate record id", (bundle) => { bundle.records[5].id = bundle.records[0].id; }, /unique record id count/],
    ["wrong geometry", (bundle) => { bundle.records[2].target.geometry.frames = 188; }, /geometry.frames/],
    ["wrong provider", (bundle) => { bundle.records[0].target.provider = "ltx_2_3"; }, /target.provider/],
    ["wrong tier", (bundle) => { bundle.records[0].target.tier = "q8"; }, /target.tier/],
    ["non-terminal row", (bundle) => { bundle.records[0].status = "gated"; }, /status/],
    ["candidate authority", (bundle) => { bundle.records[0].evidenceScope = "candidate"; }, /evidenceScope/],
    ["dirty SceneWorks", (bundle) => { bundle.records[0].repositories.sceneWorks.dirty = true; }, /sceneWorks.dirty/],
    ["wrong SceneWorks head", (bundle) => { bundle.records[0].repositories.sceneWorks.revision = "a".repeat(40); }, /sceneWorks.revision/],
    ["wrong inference pin", (bundle) => { bundle.records[0].repositories.inference.revision = "b".repeat(40); }, /inference.revision/],
    ["wrong closure", (bundle) => { bundle.records[0].repositories.inference.closureDigest = "c".repeat(64); }, /closureDigest/],
    ["wrong repository", (bundle) => { bundle.records[0].artifact.repository = "SceneWorks/lookalike"; }, /artifact.repository/],
    ["wrong artifact revision", (bundle) => { bundle.records[0].artifact.resolvedRevision = "d".repeat(40); }, /artifact.resolvedRevision/],
    ["wrong rung", (bundle) => { bundle.records[0].strategy.rung = "resident"; }, /strategy.rung/],
  ];
  for (const [label, mutate, expected] of mutations) {
    const bundle = validBundle();
    mutate(bundle);
    assert.throws(
      () => validateSc19057WanCapture({ bundle, plan: PLAN, expectedSceneWorksRevision: SCENEWORKS_REVISION }),
      expected,
      label,
    );
  }
});

test("SC-19057 rejects a reduced or mutated plan instead of redefining capture completeness", () => {
  for (const [label, mutate, expected] of [
    ["five-row plan", (plan) => { plan.providers.pop(); }, /providers.length/],
    ["reordered plan", (plan) => { plan.providers.reverse(); }, /fixture/],
    ["wrong fit role", (plan) => { plan.providers[5]._role = "held_out"; }, /_role/],
    ["wrong plan geometry", (plan) => { plan.providers[0].target.geometry.width = 800; }, /geometry.width/],
    ["wrong plan authority", (plan) => { plan.providers[0].evidenceScope = "candidate"; }, /evidenceScope/],
    ["wrong plan tier", (plan) => { plan.providers[0].target.tier = "q8"; }, /target.tier/],
  ]) {
    const plan = structuredClone(PLAN);
    mutate(plan);
    assert.throws(
      () => validateSc19057WanCapture({ bundle: validBundle(), plan, expectedSceneWorksRevision: SCENEWORKS_REVISION }),
      expected,
      label,
    );
  }
});

test("the validator CLI seals hashes and rejects an incomplete bundle before promotion", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "sc19057-validator-"));
  try {
    const input = path.join(directory, "capture.json");
    const receipt = path.join(directory, "receipt.json");
    writeFileSync(input, `${JSON.stringify(validBundle(), null, 2)}\n`);
    const baseArgs = [
      path.join(ROOT, "scripts/validate-sc19057-wan-capture.mjs"),
      "--input", input,
      "--plan", PLAN_PATH,
      "--sceneworks-revision", SCENEWORKS_REVISION,
      "--inference-revision", SC19057_INFERENCE_REVISION,
    ];
    const good = spawnSync(process.execPath, [...baseArgs, "--write-receipt", receipt], {
      cwd: ROOT,
      encoding: "utf8",
    });
    assert.equal(good.status, 0, good.stderr);
    const sealed = JSON.parse(readFileSync(receipt, "utf8"));
    assert.match(sealed.captureSha256, /^[0-9a-f]{64}$/);
    assert.match(sealed.planSha256, /^[0-9a-f]{64}$/);
    assert.equal(sealed.capturedFixtures, 6);

    const partial = validBundle();
    partial.records.pop();
    writeFileSync(input, `${JSON.stringify(partial, null, 2)}\n`);
    const bad = spawnSync(process.execPath, baseArgs, { cwd: ROOT, encoding: "utf8" });
    assert.notEqual(bad.status, 0);
    assert.match(bad.stderr, /record count must be 6/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("the SC-19057 record-terminal fitter invokes exact identity acceptance before writing", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "sc19057-fitter-boundary-"));
  try {
    const dataset = validBundle();
    dataset.records[0].artifact.repository = "SceneWorks/lookalike";
    const input = path.join(directory, "capture.json");
    const report = path.join(directory, "fit.json");
    const curves = path.join(directory, "curves.json");
    writeFileSync(input, `${JSON.stringify(dataset, null, 2)}\n`);
    const result = spawnSync(process.execPath, [
      path.join(ROOT, "scripts/fit-ltx-temporal-form.mjs"),
      "--story", "sc-19057",
      "--dataset", input,
      "--plan", PLAN_PATH,
      "--record-terminals",
      "--write", report,
      "--curve-write", curves,
      "--source-fit", "docs/generated/wan-temporal-form-fit-sc-19057.json",
    ], { cwd: ROOT, encoding: "utf8" });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /records\[0\]\.artifact\.repository/);
    assert.throws(() => readFileSync(report), /ENOENT/);
    assert.throws(() => readFileSync(curves), /ENOENT/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
