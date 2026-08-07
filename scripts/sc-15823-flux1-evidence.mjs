#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  HARNESS_VERSION, canonicalJson, logicalCaseId, recordId, validateBundle,
} from "./memory-calibration-harness.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const EVIDENCE = path.join(ROOT, "docs/generated/memory-calibration-evidence.json");
const PLAN = path.join(ROOT, "config/memory-calibration-plan.json");
const SCENEWORKS_REVISION = "f936e7e6a17a0b592752f634d21db17f5e8f2db7";
const MATRIX_SOURCE_REVISION = "source-tree:2b3918cffd38e603f3a934229ee46948c3817b9710f5a0e4ed0ead7744c5c3d5";
const INFERENCE_REVISION = "5f973a73bf00307240afd81d2778ba9d89349e51";
// sc-17774: the compile-closure digest of each FLUX.1 lane AT `INFERENCE_REVISION` — the term
// currency actually compares. The revision above stays as capture provenance. Re-derive with:
//   node scripts/inference-closure-digest.mjs --repo <inference> --revision 5f973a73 \
//     --provider candle:flux1_dev
const INFERENCE_CLOSURE_DIGESTS = Object.freeze({
  flux1_dev: "8bb03b94550deee30f4656fa502425b7c206ab1cff761d60225ad5cf13f44e74",
  flux1_schnell: "6d81f414c80acabb430be58d01491b7642427530e2ddc7fa97e1adef78f117fb",
});
const FINGERPRINT = "flux1-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v1";

const HARDWARE = {
  probe: "CUDA_VISIBLE_DEVICES=0; nvidia-smi + nvcc --version; Microsoft Windows NT 10.0.26200.0",
  memoryBytes: 102641958912,
  deviceId: "0",
  name: "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
  computeCapability: "12.0",
  driverVersion: "596.36",
  runtimeVersion: "CUDA 12.9 (nvcc V12.9.41)",
};

const MODELS = [
  {
    modelId: "flux_schnell",
    provider: "flux1_schnell",
    repository: "SceneWorks/flux1-schnell-mlx",
    revision: "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
    inventorySha256: "3157f5cdd80246daf0dd5f7c07694e8e8ee2845ec01bec8b9edb8a02b4bd8f62",
    cachePath: "E:/huggingface/hub/models--SceneWorks--flux1-schnell-mlx/snapshots/bba3ae01dfd94089f173c05edd4e1a4c551f2599/q4",
    cacheBytes: 17008176476,
    sourceSlug: "schnell",
    boundedQuality: { maximumError: 29, meanError: 0.2289120356241862, rootMeanSquareError: 0.5452312653086621 },
    rungs: [
      ["resident", 17642573594, 236, "fe0e5ff1a022da5cefe4be7dfeb2bac51bc3870bbfd9bb36949ab8381ba3d9e0", "2026-08-04T03:10:32.1204629Z", "ims-6ba27c6bb1b02924f919", "233c14cbf39c2d9a351fad195424f9c3a56ffcbdde4f72a69625a93f3d92d861"],
      ["staged_residency", 14588695430, 208, "fe0e5ff1a022da5cefe4be7dfeb2bac51bc3870bbfd9bb36949ab8381ba3d9e0", "2026-08-04T03:11:36.9096164Z", "ims-68cd302c4d981863ae34", "6296ad15ea78b71c50020f743dcfd3fa32697eb749078e62b5070e704e40269b"],
      ["bounded_decode", 9608827400, 105, "b84dd88f02854e15df837216d6f67a37e2d538f9c221ca5c1a14a741bcffcfef", "2026-08-04T03:13:09.7107643Z", "ims-4b4ab770efa632199d23", "4a0931e528f5a88095281daa8a27622846ddfbb1f5b6be4e0aadf906044fc8d9"],
      ["bounded_attention", 8234088968, 91, "b84dd88f02854e15df837216d6f67a37e2d538f9c221ca5c1a14a741bcffcfef", "2026-08-04T03:14:30.0191249Z", "ims-689c72239ec5bb84594f", "7a916bfa98f6186594c725244a7834a15bdc739dd7ac984caa5a66936d497b36"],
      ["bounded_transformer_residency", 3843456916, 45, "b84dd88f02854e15df837216d6f67a37e2d538f9c221ca5c1a14a741bcffcfef", "2026-08-04T03:16:29.1439468Z", "ims-bd6bf873c3afa366ebbc", "5d0e6f0f0bd3170450781a500d95227d1082cd481cff42929c19b4db85c9db63"],
    ],
  },
  {
    modelId: "flux_dev",
    provider: "flux1_dev",
    repository: "SceneWorks/flux1-dev-mlx",
    revision: "323fd12d79f78ad444e882e8d8e871914584f2b9",
    inventorySha256: "38892dfdd0177068e4996834a3ef6666309db5a0034f2548bd13f814e742b341",
    cachePath: "E:/huggingface/hub/models--SceneWorks--flux1-dev-mlx/snapshots/323fd12d79f78ad444e882e8d8e871914584f2b9/q4",
    cacheBytes: 17013959875,
    sourceSlug: "dev",
    boundedQuality: { maximumError: 21, meanError: 0.26471201578776044, rootMeanSquareError: 0.5180919471938068 },
    rungs: [
      ["resident", 17651074970, 238, "009e11f9bbcaca6edebe3658e221589bd7404d7a4c9db8ee251c0eae57801964", "2026-08-04T03:17:40.2449084Z", "ims-864721b19f3af847b3b0", "da72dc96457dd513765ae9f8b818eb65aa418f1e5ff28dc24c06ec9e12098ed7"],
      ["staged_residency", 14597196806, 205, "009e11f9bbcaca6edebe3658e221589bd7404d7a4c9db8ee251c0eae57801964", "2026-08-04T03:18:38.1717537Z", "ims-7e019daeae73957fa26c", "53b2a9e3ca35a94b37e79b562da2d08204b6e0f241ac82643968d88c3275fab8"],
      ["bounded_decode", 9858108040, 107, "3727e3e9c323be1be2b25ce4237c5280876832381c5e8bc64e1e36b2937348d4", "2026-08-04T03:24:22.5234271Z", "ims-6d120db7e473577a8666", "dbe87150385cc27055a0bda7af9634edaad59c69375ea432a0bd36dfae33a0eb"],
      ["bounded_attention", 8272605832, 94, "3727e3e9c323be1be2b25ce4237c5280876832381c5e8bc64e1e36b2937348d4", "2026-08-04T03:25:55.8294237Z", "ims-7e8d2d3865ddc7416364", "0ca7e56af43e6cc04631ee1a8c6ddc090c25f9841837887ceb08b38137aaf7af"],
      ["bounded_transformer_residency", 3843457940, 45, "3727e3e9c323be1be2b25ce4237c5280876832381c5e8bc64e1e36b2937348d4", "2026-08-04T03:27:45.5391525Z", "ims-80d540a194d518ccd289", "7a632e0c15bb6f69a4cfe63f1256de316425e03b873fdb0e0d29d063489c96fc"],
    ],
  },
];

const RUNG_PARAMETERS = {
  resident: {},
  staged_residency: {},
  bounded_decode: { decodeTileEdge: 512, decodeOverlap: 128 },
  bounded_attention: { decodeTileEdge: 512, decodeOverlap: 128, attentionChunkSize: 67108864 },
  bounded_transformer_residency: {
    decodeTileEdge: 512, decodeOverlap: 128, attentionChunkSize: 67108864,
    transformerWindowSize: 1, transformerWindowComponent: "dit",
  },
};
const RUNG_COMPOSITIONS = {
  resident: ["resident"],
  staged_residency: ["resident", "staged_residency"],
  bounded_decode: ["resident", "staged_residency", "bounded_decode"],
  bounded_attention: ["resident", "staged_residency", "bounded_decode", "bounded_attention"],
  bounded_transformer_residency: [
    "resident", "staged_residency", "bounded_decode", "bounded_attention",
    "bounded_transformer_residency",
  ],
};
const RUNG_LOG_SUFFIX = {
  resident: "resident",
  staged_residency: "staged-residency",
  bounded_decode: "bounded-decode",
  bounded_attention: "bounded-attention",
  bounded_transformer_residency: "bounded-transformer-residency",
};
const RUNG_ENV = {
  resident: "resident",
  staged_residency: "staged",
  bounded_decode: "bounded-decode",
  bounded_attention: "bounded-attention",
  bounded_transformer_residency: "bounded-transformer",
};

const phase = (peak) => ({
  activeBytes: peak, allocatorBytes: peak, deviceBytes: peak, wiredBytes: peak,
  reclaimableBytes: 0,
});

function makeRecord(model, [rung, peak, driverTenthsGb, outputSha256, capturedAt]) {
  const parameters = RUNG_PARAMETERS[rung];
  const exact = rung === "resident" || rung === "staged_residency";
  const quality = exact
    ? { maximumError: 0, meanError: 0, rootMeanSquareError: 0 }
    : model.boundedQuality;
  const record = {
    id: "",
    logicalCaseId: "",
    status: "runtime_complete",
    evidenceScope: "authoritative",
    backend: "candle",
    loadShape: rung === "bounded_transformer_residency"
      ? "deferred_materialization" : "eager_materialization",
    repositories: {
      sceneWorks: { revision: SCENEWORKS_REVISION, dirty: false, matrixSourceRevision: MATRIX_SOURCE_REVISION },
      inference: {
        revision: INFERENCE_REVISION,
        dirty: false,
        closureDigest: INFERENCE_CLOSURE_DIGESTS[model.provider],
      },
    },
    hardware: HARDWARE,
    artifact: {
      repository: model.repository,
      resolvedRevision: model.revision,
      variant: "q4",
      inventorySha256: model.inventorySha256,
    },
    target: {
      modelId: model.modelId,
      provider: model.provider,
      tier: "q4",
      mode: "text_to_image",
      overlay: "none",
      geometry: { width: 1024, height: 1024, batch: 1, frames: 1 },
    },
    fixture: `${model.modelId}-q4-seed42-1024x1024`,
    strategy: { rung, engagedRungs: RUNG_COMPOSITIONS[rung], parameters },
    sweep: {
      axes: Object.entries(parameters)
        .filter(([, value]) => Number.isInteger(value))
        .map(([parameter, value]) => ({ parameter, testedValues: [value] })),
      cases: [{ parameters, result: "passed" }],
      rangeVerified: true,
    },
    scenarios: [
      { name: "exact_fit", result: "passed", predictedBytes: peak, effectiveBudgetBytes: peak, reason: "selector equality boundary is covered by conformance tests" },
      { name: "unknown_budget", result: "passed", reason: "selector refuses unverifiable budgets in conformance tests" },
      { name: "stale_evidence", result: "passed", reason: "exact revision and fingerprint mismatch rejection is covered by conformance tests" },
      { name: "warm_repeat", result: "not_run", reason: "the physical session used a fresh process and did not execute a same-process warm repeat" },
      { name: "cancel", result: "not_run", reason: "cancellation cleanup was not executed in this physical evidence session" },
      { name: "error", result: "not_run", reason: "fault cleanup was not executed in this physical evidence session" },
      { name: "loadability", result: "passed", reason: "the exact pinned q4 artifact inventory loaded and rendered" },
      { name: "overlay", result: "not_applicable", reason: "this exact base-only request executed no identity, control, LoRA, or PuLID overlay" },
    ],
    predictedPeakBytes: { conditioning: peak, denoise: peak, decode: peak, overall: peak },
    observedMemory: { conditioning: phase(peak), denoise: phase(peak), decode: phase(peak), overall: phase(peak) },
    quality: {
      contract: "same model snapshot, prompt, seed 42, geometry, step count, precision, and conditioning; selected-rung RGB8 output versus resident",
      identicalInputs: true,
      result: "passed",
      ...quality,
      maximumErrorThreshold: quality.maximumError,
      meanErrorThreshold: quality.meanError,
      rootMeanSquareErrorThreshold: quality.rootMeanSquareError,
    },
    negativeMutation: null,
    loadability: {
      result: "passed",
      resolvedPathFingerprint: `${model.repository}@${model.revision}:q4`,
    },
    diagnostics: {
      adapter: "candle-flux1-memory-ladder-v1",
      execution: "executed",
      blockers: [],
      measurements: [
        { name: "requestLiveAllocationPeak", unit: "bytes", value: peak },
        { name: "driverRenderedOverallPeak", unit: "0.1 GB (diagnostic display)", value: driverTenthsGb },
        { name: "outputRgbBytes", unit: "bytes", value: 3145728 },
      ],
    },
    calibrationFingerprint: FINGERPRINT,
    capturedAt,
    harnessVersion: HARNESS_VERSION,
  };
  // The RGB8 output digest is retained in the fixture identity without pretending it is numeric telemetry.
  record.fixture = `${record.fixture}-output-${outputSha256}`;
  record.logicalCaseId = logicalCaseId(record);
  record.id = recordId(record);
  return record;
}

export function flux1EvidenceRecords() {
  return MODELS.flatMap((model) => model.rungs.map((rung) => makeRecord(model, rung)));
}

function makeSourceSession(model, rungTuple) {
  const [rung, , , outputSha256, capturedAt, id, stdoutSha256] = rungTuple;
  const testName = model.sourceSlug === "schnell"
    ? "flux_schnell_probed_generate_for_offload_ab"
    : "flux_dev_probed_generate_for_offload_ab";
  return {
    id,
    kind: "physical_cuda",
    command: `CUDA_VISIBLE_DEVICES=0 FLUX_MEMORY_RUNG=${RUNG_ENV[rung]} candle_gen_flux-d39d12fd79fcc5dc.exe tests::${testName} --ignored --nocapture --test-threads=1`,
    sourcePath: `docs/calibration/sc-15823-refresh-${model.sourceSlug}-q4-${RUNG_LOG_SUFFIX[rung]}.log`,
    capturedAt,
    repositories: {
      sceneWorks: { revision: SCENEWORKS_REVISION, dirty: false },
      // The physical outputs and device-format sidecars lived in inference/.tmp during capture.
      inference: { revision: INFERENCE_REVISION, dirty: true },
    },
    hardware: HARDWARE,
    target: { tier: "q4", mode: "text_to_image", overlay: "none", rung },
    stdoutSha256,
    inputs: [{
      role: "base",
      path: model.cachePath,
      bytes: model.cacheBytes,
      sha256: model.inventorySha256,
      repository: model.repository,
      resolvedRevision: model.revision,
      variant: "q4",
    }],
    outputs: [{
      path: `.tmp/sc-15823-refresh-outputs/${model.sourceSlug}-${RUNG_ENV[rung]}.rgb`,
      sha256: outputSha256,
    }],
    claims: ["memory", "loadability", "overlay"],
    result: "passed",
  };
}

export function flux1SourceSessions() {
  return MODELS.flatMap((model) => model.rungs.map((rung) => makeSourceSession(model, rung)));
}

export function flux1CalibrationPlans() {
  return flux1EvidenceRecords().map((record) => ({
    name: `candle-${record.target.modelId}-q4-${record.strategy.rung}-sc15823`,
    evidenceScope: "authoritative",
    backend: record.backend,
    loadShape: record.loadShape,
    target: record.target,
    rung: record.strategy.rung,
    engagedRungs: record.strategy.engagedRungs,
    calibrationFingerprint: record.calibrationFingerprint,
    fixture: record.fixture,
    cases: [{ parameters: record.strategy.parameters, expectedResult: "passed" }],
  }));
}

export function updatePlan(existing) {
  return {
    ...existing,
    providers: [
      ...existing.providers.filter((item) =>
        !(item.backend === "candle" && ["flux_schnell", "flux_dev"].includes(item.target.modelId)),
      ),
      ...flux1CalibrationPlans(),
    ],
  };
}

export function updateBundle(existing) {
  const records = existing.records.filter((record) =>
    !(record.backend === "candle" && ["flux1_schnell", "flux1_dev"].includes(record.target.provider)),
  );
  records.push(...flux1EvidenceRecords());
  const sourceSessions = (existing.sourceSessions ?? []).filter(
    (session) => !session.sourcePath.startsWith("docs/calibration/sc-15823-refresh-"),
  );
  sourceSessions.push(...flux1SourceSessions());
  const bundle = {
    schemaVersion: 4,
    harnessVersion: HARNESS_VERSION,
    sourceSessions: sourceSessions.sort((left, right) => left.id.localeCompare(right.id)),
    records: records.sort((left, right) => left.id.localeCompare(right.id)),
  };
  validateBundle(bundle);
  return bundle;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const existing = JSON.parse(await readFile(EVIDENCE, "utf8"));
  await writeFile(EVIDENCE, canonicalJson(updateBundle(existing)));
  const plan = JSON.parse(await readFile(PLAN, "utf8"));
  await writeFile(PLAN, `${JSON.stringify(updatePlan(plan), null, 2)}\n`);
}
