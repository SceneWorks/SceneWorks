import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";

let body = "";
for await (const chunk of process.stdin) body += chunk;
const request = JSON.parse(body);

if (request.action === "probe") {
  process.stdout.write(JSON.stringify({
    hardware: {
      probe: "gated canonical-batch fixture probe",
      memoryBytes: 51539607552,
      deviceId: "fixture:0",
      name: "Fixture CUDA",
      computeCapability: "9.0",
      driverVersion: "999.1",
      runtimeVersion: "12.8",
    },
  }));
  process.exit(0);
}

const canonicalRungs = [
  "resident",
  "staged_residency",
  "bounded_decode",
  "bounded_attention",
  "bounded_transformer_residency",
];
if (request.action === "run_batch") {
  const actualRungs = request.planned.map((planned) => planned.strategy.rung);
  if (JSON.stringify(actualRungs) !== JSON.stringify(canonicalRungs)) {
    process.stderr.write(`fixture refuses partial rung batch ${JSON.stringify(actualRungs)}`);
    process.exit(1);
  }
}

const stateFile = process.argv[2];
const failAfter = process.argv[3] === undefined ? null : Number(process.argv[3]);
let capturedAt = "2026-08-03T12:00:00Z";
if (stateFile) {
  let state;
  try {
    state = JSON.parse(await readFile(stateFile, "utf8"));
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    state = { successfulInvocations: 0 };
  }
  if (Number.isInteger(failAfter) && state.successfulInvocations >= failAfter) {
    process.stderr.write(`fixture fails after ${failAfter} successful invocations`);
    process.exit(1);
  }
  state.successfulInvocations += 1;
  capturedAt = `2026-08-03T12:00:${String(state.successfulInvocations).padStart(2, "0")}Z`;
  await writeFile(stateFile, JSON.stringify(state));
}

const phase = (value) => ({
  activeBytes: value,
  allocatorBytes: value,
  reclaimableBytes: 0,
});
const fragment = (planned) => {
  const parameters = planned.strategy.parameters;
  return {
    status: "gated",
    strategy: planned.strategy,
    loadShape: planned.loadShape,
    artifact: {
      repository: "SceneWorks/qwen-image-mlx",
      resolvedRevision: "c".repeat(40),
      variant: "q4",
    },
    sweep: {
      axes: Object.entries(parameters).map(([parameter, value]) => ({
        parameter,
        testedValues: [value],
      })),
      cases: [{ parameters, result: "passed" }],
      rangeVerified: false,
    },
    scenarios: [
      "exact_fit", "unknown_budget", "stale_evidence", "warm_repeat",
      "cancel", "error", "loadability",
    ].map((name) => ({ name, result: "not_run", reason: "fixture remains gated" })).concat({
      name: "overlay",
      result: "not_applicable",
      reason: "fixture has no overlay",
    }),
    predictedPeakBytes: null,
    observedMemory: {
      conditioning: phase(100),
      denoise: phase(200),
      decode: phase(150),
      overall: phase(200),
    },
    quality: { result: "not_run" },
    negativeMutation: null,
    loadability: { result: "passed", resolvedPathFingerprint: "fixture@resolved:q4" },
    diagnostics: {
      adapter: "gated-canonical-batch-fixture",
      execution: "executed",
      blockers: ["fixture remains gated"],
      measurements: [
        { name: "conditioningDevicePeakDelta", value: 100, unit: "bytes" },
        { name: "denoiseDevicePeakDelta", value: 200, unit: "bytes" },
        { name: "decodeDevicePeakDelta", value: 150, unit: "bytes" },
        { name: "overallDevicePeakDelta", value: 200, unit: "bytes" },
      ],
    },
    capturedAt,
  };
};

if (request.action === "run_batch") {
  process.stdout.write(JSON.stringify({
    modelLoads: 1,
    fragments: request.planned.map(fragment),
  }));
} else if (request.action === "run") {
  process.stdout.write(JSON.stringify(fragment(request.planned)));
} else {
  process.stderr.write(`unsupported fixture action ${JSON.stringify(request.action)}`);
  process.exit(1);
}
