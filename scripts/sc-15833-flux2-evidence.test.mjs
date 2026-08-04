import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { validateRecord } from "./memory-calibration-harness.mjs";
import {
  flux2CalibrationPlans,
  ingestFlux2Capture,
  updateBundle,
  updatePlan,
} from "./sc-15833-flux2-evidence.mjs";

const SCENEWORKS_REVISION = "1".repeat(40);
const INFERENCE_REVISION = "2".repeat(40);
const MODEL_REVISION = "3".repeat(40);
const MODEL_INVENTORY = "4".repeat(64);
const CONTROL_REVISION = "5".repeat(40);
const CONTROL_SHA = "6".repeat(64);
const MATRIX_SOURCE = `source-tree:${"7".repeat(64)}`;
const FINGERPRINT = "flux2-dev-cuda-staged-host-full-edge-decode-bounded-attention-device-format-blocks-v2";
const OUTPUT = Buffer.alloc(1024 * 1024 * 3, 17);
const OUTPUT_SHA = createHash("sha256").update(OUTPUT).digest("hex");
const REPOSITORIES = {
  sceneWorks: { revision: SCENEWORKS_REVISION, dirty: false, matrixSourceRevision: MATRIX_SOURCE },
  inference: { revision: INFERENCE_REVISION, dirty: false },
};
const HARDWARE = {
  probe: "CUDA_VISIBLE_DEVICES=0; nvidia-smi + nvcc --version",
  memoryBytes: 102641958912,
  deviceId: "0",
  name: "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
  computeCapability: "12.0",
  driverVersion: "596.36",
  runtimeVersion: "CUDA 12.9 (nvcc V12.9.41)",
};
const BASE_INPUT = {
  role: "base",
  path: "E:/huggingface/sceneworks-staging/sc15833-flux2-dev-q4",
  bytes: 33582076196,
  sha256: MODEL_INVENTORY,
  repository: "SceneWorks/flux2-dev-mlx",
  resolvedRevision: MODEL_REVISION,
  variant: "q4",
};
const CONTROL_INPUT = {
  role: "control",
  path: "E:/huggingface/sceneworks-staging/sc15833-flux2-control/control.safetensors",
  bytes: 8232506680,
  sha256: CONTROL_SHA,
  repository: "xiaozaa/catvton-flux2-control",
  resolvedRevision: CONTROL_REVISION,
  variant: "union-2602",
};

const RUNGS = [
  "resident",
  "staged_residency",
  "bounded_decode",
  "bounded_attention",
  "bounded_transformer_residency",
];

const COMPOSITIONS = {
  resident: ["resident"],
  staged_residency: ["resident", "staged_residency"],
  bounded_decode: ["resident", "staged_residency", "bounded_decode"],
  bounded_attention: ["resident", "staged_residency", "bounded_decode", "bounded_attention"],
  bounded_transformer_residency: [
    "resident", "staged_residency", "bounded_decode", "bounded_attention",
    "bounded_transformer_residency",
  ],
};

const PARAMS = {
  resident: [null, null, null, null, null],
  staged_residency: [null, null, null, null, null],
  bounded_decode: [1024, 1, null, null, null],
  bounded_attention: [1024, 1, 67108864, null, null],
  bounded_transformer_residency: [1024, 1, 67108864, 1, "dit"],
};

const STRATEGY_DEBUG = {
  staged_residency: "StagedResidency",
  bounded_decode: "BoundedDecode",
  bounded_attention: "BoundedAttention",
  bounded_transformer_residency: "BoundedTransformerResidency",
};

function memoryRecord(rung) {
  const [decodeTile, overlap, attention, window, component] = PARAMS[rung];
  const bounded = rung.startsWith("bounded_");
  return {
    schema_version: 1,
    key: {
      resolved_route: "flux2_dev",
      backend: "candle",
      tier: { precision: "bf16", quant: "q4", component_precision_floors: [] },
      load_shape: "deferred_materialization",
      mode: "text_to_image",
      overlay: null,
      geometry: { width: 1024, height: 1024, batch: 1, frames: 1, reference_count: 0 },
      strategy: rung,
      engaged_composition: COMPOSITIONS[rung],
      parameters: {
        decode_tile_edge: decodeTile,
        decode_overlap: overlap,
        attention_chunk_size: attention,
        transformer_window_size: window,
        transformer_window_component: component,
      },
    },
    declared_calibration: { abi: 3, fingerprint: FINGERPRINT, load_shape: "deferred_materialization" },
    observed_calibration: { abi: 3, fingerprint: FINGERPRINT, load_shape: "deferred_materialization" },
    predicted_peak_bytes: 20_000_000_000 - RUNGS.indexOf(rung) * 1_000_000_000,
    observed_peak_bytes: 20_000_000_000 - RUNGS.indexOf(rung) * 1_000_000_000,
    inference_revision: INFERENCE_REVISION,
    sceneworks_revision: SCENEWORKS_REVISION,
    model_revision: MODEL_REVISION,
    model_inventory_sha256: MODEL_INVENTORY,
    harness_version: "candle-flux2-memory-ladder-v1",
    output_sha256: OUTPUT_SHA,
    parity: bounded
      ? { kind: "tolerance", metric: "rgb8_max_abs_error", maximum_error: 2 }
      : { kind: "exact" },
    parity_result: { kind: rung === "resident" ? "not_run" : "passed" },
  };
}

function wrappedJson(value) {
  const json = JSON.stringify(value);
  return json.match(/.{1,71}/g).join("\n");
}

function sourceHeader(capturedAt, inputs) {
  return `SC15833_SOURCE_V1 ${wrappedJson({
    schema_version: 1,
    captured_at: capturedAt,
    repositories: REPOSITORIES,
    hardware: HARDWARE,
    inputs,
  })}\n`;
}

function baseLog(rung, capturedAt, overrides = {}) {
  const record = structuredClone(memoryRecord(rung));
  Object.assign(record, overrides.record ?? {});
  let diagnostic = "";
  if (rung !== "resident") {
    const bounded = rung.startsWith("bounded_");
    const maximum = overrides.maximumError ?? (bounded ? 2 : 0);
    const changed = maximum === 0 ? 0 : 0.040744145711;
    const mean = maximum === 0 ? 0 : 0.040746370951;
    const rmse = maximum === 0 ? 0 : 0.201868326965;
    const psnr = maximum === 0 ? "inf" : "62.029439934468";
    diagnostic = `MEMORY_PARITY_DIAGNOSTIC strategy=${STRATEGY_DEBUG[rung]} reference=docs/calibration/sc-15833/base-q4-resident.rgb changed_fraction=${changed} max_abs=${maximum} mean_abs=${mean} rmse=${rmse} psnr_db=${psnr}\n`;
  }
  return Buffer.from(
    `${sourceHeader(capturedAt, [BASE_INPUT])}running 1 test\nMEMORY_EVIDENCE_V1 ${wrappedJson(record)}\n${diagnostic}` +
    "test tests::flux2_dev_probed_generate_for_offload_ab ... ok\n" +
    "test result: ok. 1 passed; 0 failed\n",
  );
}

function routeLog(route, capturedAt) {
  const testName = route === "edit"
    ? "flux2_dev_edit_candle_gpu_smoke"
    : "flux2_dev_control_candle_gpu_smoke";
  return Buffer.from(
    `${sourceHeader(capturedAt, route === "control" ? [BASE_INPUT, CONTROL_INPUT] : [BASE_INPUT])}running 1 test\n` +
    `[smoke] dev ${route} 512x512 std ${route === "edit" ? "80.28" : "31.82"} -> output.png\n` +
    `[smoke] DONE: flux2_dev ${route} (candle) coherent\n` +
    `test flux2_dev_gpu_smoke::${testName} ... ok\n` +
    "test result: ok. 1 passed; 0 failed\n",
  );
}

function supportLog(kind, capturedAt) {
  const marker = kind === "lifecycle"
    ? {
      kind,
      result: "passed",
      warm_repeat: true,
      cancel_cleanup: true,
      cancel_warm_follow_up: true,
      error_cleanup: true,
      error_warm_follow_up: true,
    }
    : { kind, result: "passed", measured: true, maximum_error: 255, mean_error: 13.25 };
  return Buffer.from(
    `${sourceHeader(capturedAt, [BASE_INPUT])}SC15833_VALIDATION_V1 ${wrappedJson(marker)}\n` +
    `test sc15833_${kind} ... ok\ntest result: ok. 1 passed; 0 failed\n`,
  );
}

function fixture({ promotionIntent = "complete", support = true } = {}) {
  const files = new Map();
  const baseSessions = RUNGS.map((rung, index) => {
    const sourcePath = `docs/calibration/sc-15833/base-q4-${rung}.log`;
    const outputPath = `docs/calibration/sc-15833/base-q4-${rung}.rgb`;
    const capturedAt = `2026-08-04T04:0${index}:00Z`;
    files.set(sourcePath, baseLog(rung, capturedAt));
    files.set(outputPath, OUTPUT);
    return {
      rung,
      command: `CUDA_VISIBLE_DEVICES=0 FLUX2_MEMORY_RUNG=${rung} cargo test --release --ignored`,
      sourcePath,
      capturedAt,
      outputPath,
    };
  });
  const routeSessions = ["edit", "control"].map((route, index) => {
    const sourcePath = `docs/calibration/sc-15833/${route}-q4-resident.log`;
    const outputPath = `docs/calibration/sc-15833/${route}-q4-resident.png`;
    const capturedAt = `2026-08-04T04:1${index}:00Z`;
    files.set(sourcePath, routeLog(route, capturedAt));
    files.set(outputPath, Buffer.from(`${route}-png`));
    return {
      route,
      command: `CUDA_VISIBLE_DEVICES=0 cargo test --release ${route}_gpu_smoke -- --ignored`,
      sourcePath,
      capturedAt,
      outputPath,
    };
  });
  const supportSessions = support ? ["lifecycle", "negative_mutation"].map((kind, index) => {
    const sourcePath = `docs/calibration/sc-15833/${kind}.log`;
    const capturedAt = `2026-08-04T04:2${index}:00Z`;
    files.set(sourcePath, supportLog(kind, capturedAt));
    return {
      kind,
      command: `cargo test sc15833_${kind} -- --nocapture`,
      sourcePath,
      capturedAt,
    };
  }) : [];
  const excludedPath = "target/sc15833-evidence/dirty_experimental_decode_1024_1.log";
  files.set(excludedPath, Buffer.from("diagnostic only; repositories intentionally dirty"));
  const capture = {
    schemaVersion: 1,
    story: "SC-15833",
    promotionIntent,
    repositories: structuredClone(REPOSITORIES),
    hardware: structuredClone(HARDWARE),
    calibrationAbi: 3,
    calibrationFingerprint: FINGERPRINT,
    baseInput: structuredClone(BASE_INPUT),
    controlInput: structuredClone(CONTROL_INPUT),
    baseSessions,
    routeSessions,
    supportSessions,
    excludedAttempts: [{
      sourcePath: excludedPath,
      reason: "diagnostic sweep ran from a dirty experimental inference tree and is never promotion evidence",
    }],
  };
  const reader = async (relative) => {
    const value = files.get(relative);
    if (!value) throw new Error(`missing fixture ${relative}`);
    return value;
  };
  return { capture, files, reader };
}

test("SC-15833 ingests five strict base rungs plus edit/control and support sessions", async () => {
  const { capture, reader } = fixture();
  const ingestion = await ingestFlux2Capture(capture, { reader });
  assert.equal(ingestion.report.promotionEligible, true);
  assert.deepEqual(ingestion.report.blockers, []);
  assert.equal(ingestion.records.length, 5);
  assert.equal(ingestion.sourceSessions.length, 9);
  assert.equal(ingestion.report.acceptedSessions.length, 9);
  assert.equal(ingestion.report.excludedAttempts.length, 1);
  assert.match(ingestion.report.excludedAttempts[0].reason, /dirty experimental/);
  for (const record of ingestion.records) {
    validateRecord(record);
    assert.equal(record.status, "complete");
    assert.equal(record.repositories.inference.revision, INFERENCE_REVISION);
    assert.equal(record.repositories.sceneWorks.revision, SCENEWORKS_REVISION);
    assert.equal(record.target.provider, "flux2_dev");
  }
  assert.deepEqual(
    ingestion.report.baseRungs.map(({ rung }) => rung),
    RUNGS,
  );
  assert.equal(ingestion.report.baseRungs[1].parity.kind, "exact");
  assert.equal(ingestion.report.baseRungs[1].parityResult.kind, "passed");
  for (const rung of ingestion.report.baseRungs.slice(2)) {
    assert.deepEqual(rung.parity, {
      kind: "tolerance", metric: "rgb8_max_abs_error", maximum_error: 2,
    });
    assert.equal(rung.parityResult.kind, "passed");
    assert.equal(rung.diagnostics.maximumError, 2);
    assert.ok(rung.diagnostics.meanError > 0);
    assert.ok(rung.diagnostics.rootMeanSquareError > 0);
    assert.ok(rung.diagnostics.psnrDb > 60);
  }
  assert.deepEqual(
    ingestion.report.routes.map(({ route }) => route).sort(),
    ["control", "edit"],
  );
});

test("SC-15833 report-only and incomplete validation can never update promotion artifacts", async () => {
  const reportOnly = fixture({ promotionIntent: "report_only" });
  const reported = await ingestFlux2Capture(reportOnly.capture, { reader: reportOnly.reader });
  assert.equal(reported.report.promotionEligible, false);
  assert.equal(reported.records.length, 0);
  assert.match(reported.report.blockers.join("\n"), /report_only/);

  const existing = JSON.parse(await readFile(new URL("../docs/generated/memory-calibration-evidence.json", import.meta.url)));
  assert.throws(() => updateBundle(existing, reported), /refusing evidence update/);

  const incomplete = fixture({ support: false });
  const blocked = await ingestFlux2Capture(incomplete.capture, { reader: incomplete.reader });
  assert.equal(blocked.report.promotionEligible, false);
  assert.equal(blocked.records.length, 0);
  assert.match(blocked.report.blockers.join("\n"), /lifecycle/);
  assert.match(blocked.report.blockers.join("\n"), /negative-mutation/);
});

test("SC-15833 rejects staged drift and bounded diagnostics above provider maxAbs", async () => {
  const staged = fixture();
  staged.files.set(
    "docs/calibration/sc-15833/base-q4-staged_residency.log",
    baseLog("staged_residency", "2026-08-04T04:01:00Z", { maximumError: 1 }),
  );
  await assert.rejects(
    ingestFlux2Capture(staged.capture, { reader: staged.reader }),
    /staged output is not byte-exact/,
  );

  const bounded = fixture();
  bounded.files.set(
    "docs/calibration/sc-15833/base-q4-bounded_decode.log",
    baseLog("bounded_decode", "2026-08-04T04:02:00Z", { maximumError: 3 }),
  );
  await assert.rejects(
    ingestFlux2Capture(bounded.capture, { reader: bounded.reader }),
    /exceeds maxAbs 2/,
  );

  const stagedDigest = fixture();
  const differentOutput = Buffer.alloc(1024 * 1024 * 3, 18);
  const differentSha = createHash("sha256").update(differentOutput).digest("hex");
  stagedDigest.files.set(
    "docs/calibration/sc-15833/base-q4-staged_residency.log",
    baseLog("staged_residency", "2026-08-04T04:01:00Z", {
      record: { output_sha256: differentSha },
    }),
  );
  stagedDigest.files.set(
    "docs/calibration/sc-15833/base-q4-staged_residency.rgb",
    differentOutput,
  );
  const blocked = await ingestFlux2Capture(stagedDigest.capture, { reader: stagedDigest.reader });
  assert.equal(blocked.report.promotionEligible, false);
  assert.match(blocked.report.blockers.join("\n"), /not byte-identical to the resident source output/);
});

test("SC-15833 rejects dirty naming, dirty repositories, malformed provenance, and accepted/excluded overlap", async () => {
  const named = fixture();
  const session = named.capture.baseSessions[0];
  const oldPath = session.sourcePath;
  session.sourcePath = "docs/calibration/sc-15833/dirty-experimental-resident.log";
  named.files.set(session.sourcePath, named.files.get(oldPath));
  await assert.rejects(
    ingestFlux2Capture(named.capture, { reader: named.reader }),
    /excluded from authoritative ingestion/,
  );

  const dirty = fixture();
  dirty.capture.repositories.inference.dirty = true;
  await assert.rejects(ingestFlux2Capture(dirty.capture, { reader: dirty.reader }), /must be clean/);

  const zero = fixture();
  zero.capture.repositories.inference.revision = "0".repeat(40);
  await assert.rejects(ingestFlux2Capture(zero.capture, { reader: zero.reader }), /nonzero lowercase Git SHA/);

  const hardware = fixture();
  hardware.capture.hardware.driverVersion = "unexpected-driver";
  await assert.rejects(
    ingestFlux2Capture(hardware.capture, { reader: hardware.reader }),
    /hardware provenance mismatch/,
  );

  const absolute = fixture();
  absolute.capture.baseSessions[0].sourcePath = "C:/temp/resident.log";
  await assert.rejects(
    ingestFlux2Capture(absolute.capture, { reader: absolute.reader }),
    /must stay inside the repository/,
  );

  const overlap = fixture();
  overlap.capture.excludedAttempts[0].sourcePath = overlap.capture.baseSessions[0].sourcePath;
  await assert.rejects(ingestFlux2Capture(overlap.capture, { reader: overlap.reader }), /both accepted and excluded/);
});

test("SC-15833 bundle and plan updates preserve all unrelated evidence structurally", async () => {
  const { capture, reader } = fixture();
  const ingestion = await ingestFlux2Capture(capture, { reader });
  const existing = JSON.parse(await readFile(new URL("../docs/generated/memory-calibration-evidence.json", import.meta.url)));
  const unrelatedRecords = existing.records.filter((record) =>
    !(record.backend === "candle" && record.target.provider === "flux2_dev"),
  );
  for (const provider of ["qwen_image", "z_image_turbo"]) {
    assert.ok(
      unrelatedRecords.some((record) => record.target.provider === provider),
      `fixture bundle must exercise structural preservation for ${provider}`,
    );
  }
  const unrelatedSessions = (existing.sourceSessions ?? []).filter(
    (session) => !session.sourcePath.startsWith("docs/calibration/sc-15833/"),
  );
  const updated = updateBundle(existing, ingestion);
  for (const record of unrelatedRecords) {
    assert.deepEqual(updated.records.find(({ id }) => id === record.id), record);
  }
  for (const session of unrelatedSessions) {
    assert.deepEqual(updated.sourceSessions.find(({ id }) => id === session.id), session);
  }
  assert.equal(updated.records.filter((record) => record.target.provider === "flux2_dev").length, 5);
  assert.equal(
    updated.sourceSessions.filter((session) => session.sourcePath.startsWith("docs/calibration/sc-15833/")).length,
    9,
  );

  const plan = JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url)));
  const updatedPlan = updatePlan(plan, ingestion);
  const plans = flux2CalibrationPlans(ingestion.records);
  const unrelatedPlanItems = plan.providers.filter((item) => item.target.provider !== "flux2_dev");
  assert.deepEqual(
    updatedPlan.providers.filter((item) => item.target.provider !== "flux2_dev"),
    unrelatedPlanItems,
  );
  for (const provider of ["qwen_image", "qwen_image_edit", "z_image", "z_image_turbo"]) {
    assert.ok(
      unrelatedPlanItems.some((item) => item.target.provider === provider),
      `fixture plan must exercise structural preservation for ${provider}`,
    );
  }
  assert.equal(plans.length, 5);
  assert.equal(updatedPlan.providers.filter((item) => item.target.provider === "flux2_dev").length, 5);
  assert.deepEqual(plans.map(({ rung }) => rung), RUNGS);
});
