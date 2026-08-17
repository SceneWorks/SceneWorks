import process from "node:process";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const captureDir = process.argv[2];
const sourcePrefix = process.argv[3];
const outputDir = process.argv[4] ?? captureDir;
const captureMutation = process.argv[5] ?? null;

let body = "";
for await (const chunk of process.stdin) body += chunk;
const request = JSON.parse(body);
if (request.action === "assess_batch") {
  process.stdout.write(JSON.stringify({
    verdict: "eligible_for_measurement",
    reason: "fixture provider exposes one compatible load shape",
  }));
  process.exit(0);
}
if (request.action === "probe") {
  process.stdout.write(JSON.stringify({
    hardware: captureDir ? {
      probe: "executable physical MLX fixture probe",
      memoryBytes: 137438953472,
      model: "Mac17,6",
      chip: "Apple M5 Max",
      osVersion: "26.5.2",
      metalDevice: "Apple M5 Max",
      mlxMemoryLimitBytes: 130567005798,
      wiredLimitBytes: 87044670532,
    } : {
      probe: "executable fixture probe",
      memoryBytes: 51539607552,
      deviceId: "fixture:0",
      name: "Fixture CUDA",
      computeCapability: "9.0",
      driverVersion: "999.1",
      runtimeVersion: "12.8"
    }
  }));
  process.exit(0);
}
const phase = (value) => ({
  activeBytes: value, allocatorBytes: value + 10, reclaimableBytes: 10
});
const fragment = async (planned) => {
  const p = planned.strategy.parameters;
  const result = {
    status: request.repositories.sceneWorks.dirty || request.repositories.inference.dirty ? "gated" : "complete",
    strategy: planned.strategy,
    loadShape: planned.loadShape,
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
      parameters: { decodeTileEdge: 256, decodeOverlap: 32 }, measured: true,
      result: "failed_as_expected", maximumError: 0.09, meanError: 0.02
    },
    loadability: { result: "passed", resolvedPathFingerprint: "fixture@resolved:q4" },
    capturedAt: "2026-07-28T12:00:00Z"
  };
  if (captureDir) {
    const receiptDir = path.join(outputDir, ...sourcePrefix.split("/"));
    await mkdir(receiptDir, { recursive: true });
    const persistRgb = async (role, fill) => {
      const bytes = Buffer.alloc(
        planned.target.geometry.width * planned.target.geometry.height * 3,
        fill,
      );
      const sha256 = createHash("sha256").update(bytes).digest("hex");
      const fileName = `${planned.logicalCaseId}-${role}-${planned.target.geometry.width}x${
        planned.target.geometry.height}-${sha256}.rgb`;
      const localPath = path.join(receiptDir, fileName);
      await writeFile(localPath, bytes);
      return {
        role,
        path: `${sourcePrefix}/${fileName}`,
        localPath,
        sha256,
        bytes: bytes.length,
      };
    };
    const selected = await persistRgb("selected_rgb", 0x31);
    const reference = await persistRgb("reference_rgb", 0x52);
    if (captureMutation === "tamper-after-attest") {
      await writeFile(selected.localPath, Buffer.alloc(selected.bytes, 0xff));
    }
    result.sourceCapture = {
      kind: "physical_mlx",
      inputs: [{
        role: "base",
        path: "/fixture/q4",
        bytes: 1234,
        sha256: "d".repeat(64),
        repository: "SceneWorks/fixture",
        resolvedRevision: "c".repeat(40),
        variant: "q4",
      }],
      outputs: [selected, reference],
      claims: ["memory", "quality", "negative_mutation", "lifecycle", "loadability", "overlay"],
    };
  }
  return result;
};
if (request.action === "run_batch") {
  process.stdout.write(JSON.stringify({
    modelLoads: 1,
    fragments: await Promise.all(request.planned.map(fragment)),
  }));
} else {
  process.stdout.write(JSON.stringify(await fragment(request.planned)));
}
