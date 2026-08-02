import process from "node:process";

let body = "";
for await (const chunk of process.stdin) body += chunk;
const request = JSON.parse(body);
let response;
if (request.action === "probe") {
  response = {
    hardware: {
      probe: "fragmented executable fixture probe",
      memoryBytes: 51539607552,
      deviceId: "fixture:0",
      name: "Fixture CUDA",
      computeCapability: "9.0",
      driverVersion: "999.1",
      runtimeVersion: "12.8"
    }
  };
} else {
  const p = request.planned.strategy.parameters;
  const phase = (value) => ({
    activeBytes: value, allocatorBytes: value + 10, deviceBytes: value + 20,
    wiredBytes: value + 30, reclaimableBytes: 0
  });
  const negative = request.planned.expectedResult === "failed";
  response = {
    status: negative
      ? "negative_complete"
      : request.repositories.sceneWorks.dirty || request.repositories.inference.dirty
        ? "gated"
        : "complete",
    strategy: request.planned.strategy,
    loadShape: request.planned.loadShape,
    artifact: { repository: "SceneWorks/fixture", resolvedRevision: "cccccccccccccccccccccccccccccccccccccccc", variant: "q4" },
    sweep: {
      axes: [{ parameter: "decodeTileEdge", testedValues: [384, 512] }],
      cases: [
        { parameters: { ...p, decodeTileEdge: 384 }, result: "passed" },
        { parameters: { ...p, decodeTileEdge: 512 }, result: "passed" }
      ],
      rangeVerified: true
    },
    scenarios: [
      { name: "exact_fit", result: "passed", predictedBytes: 200, effectiveBudgetBytes: 200 },
      { name: "unknown_budget", result: "passed" },
      { name: "stale_evidence", result: "passed" },
      { name: "warm_repeat", result: "passed" },
      { name: "cancel", result: "passed", cleanupVerified: true, warmFollowUpPassed: true },
      { name: "error", result: "passed", cleanupVerified: true, warmFollowUpPassed: true },
      { name: "loadability", result: "passed" },
      { name: "overlay", result: "not_applicable", reason: "fixture has no overlay" }
    ],
    predictedPeakBytes: { conditioning: 100, denoise: 200, decode: 150, overall: 200 },
    observedMemory: { conditioning: phase(100), denoise: phase(200), decode: phase(150), overall: phase(200) },
    quality: {
      contract: "tolerance", identicalLatents: true, result: "passed",
      maximumError: 0.01, meanError: 0.001,
      maximumErrorThreshold: 0.08, meanErrorThreshold: 0.01
    },
    negativeMutation: {
      parameters: negative ? p : { decodeTileEdge: 256, decodeOverlap: 32 }, measured: true,
      result: "failed_as_expected", maximumError: 0.09, meanError: 0.02
    },
    loadability: { result: "passed", resolvedPathFingerprint: "fixture@resolved:q4" },
    capturedAt: "2026-07-28T12:00:00Z"
  };
}
const encoded = JSON.stringify(response);
const split = Math.max(1, Math.floor(encoded.length / 3));
process.stdout.write(encoded.slice(0, split));
setTimeout(() => {
  process.stdout.write(encoded.slice(split, split * 2));
  setTimeout(() => process.stdout.end(encoded.slice(split * 2)), 5);
}, 5);
