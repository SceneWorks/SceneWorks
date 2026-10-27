import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { buildEndpointConfig } from "../../scripts/package-sdxl-cuda-endpoints.mjs";

// Build-time projection of the same retained pair the worker uses. No capture paths, artifact
// inventories or device identifiers enter the browser, and Vite's filesystem allowlist stays closed.
const repositoryRoot = pathToFileURL(`${resolve(dirname(fileURLToPath(import.meta.url)), "../..")}/`);
export function loadSdxlCudaMemoryProjection(root = repositoryRoot) {
  try {
    const read = (path) => readFileSync(new URL(path, root), "utf8");
    const config = JSON.parse(read("config/sdxl-cuda-memory-endpoints.json"));
    if (config.schemaVersion !== 1 || !config.pair) return null;
    const { resident, sequential, summary: receipt } = config.pair;
    const summaryRaw = read(receipt.path);
    const sha256 = (raw) => createHash("sha256").update(raw).digest("hex");
    if (sha256(summaryRaw) !== receipt.sha256) return null;
    const summary = JSON.parse(summaryRaw);
    const bundles = { resident: read(resident.source.path), sequential: read(sequential.source.path) };
    const rebuilt = buildEndpointConfig(summaryRaw, bundles, resident.source.loaderClosureDigest);
    if (JSON.stringify(rebuilt) !== JSON.stringify(config)) return null;
    const optimized = summary.endpoints.find((endpoint) => endpoint.endpoint === "sequential");
    const projectEndpoint = (endpoint) => ({
      peakBytes: endpoint.overallAllocatorEnvelopeBytes,
      calibrationFingerprint: endpoint.source.calibrationFingerprint,
      recordId: endpoint.source.recordId,
      sourceSha256: endpoint.source.sha256,
    });
    return {
      schemaVersion: 1,
      modelId: sequential.modelId,
      backend: sequential.backend,
      tier: sequential.tier,
      mode: sequential.mode,
      geometry: { ...optimized.planned.target.geometry, referenceCount: sequential.referenceCount },
      resident: projectEndpoint(resident),
      optimized: { ...projectEndpoint(sequential), parameters: optimized.planned.strategy.parameters },
      settings: { ...summary.resolvedDefaults, count: optimized.settings.count },
      // Checked mirrors of the worker's accounting allowance and dedicated-VRAM reserve. These
      // are policy, not measured peak bytes or a measured same-cell repeat spread.
      recaptureSpread: 0.02,
      reserveGb: 2,
      physicalEightGiBDeviceValidated: summary.physicalEightGiBDeviceValidated,
    };
  } catch {
    // Absent or malformed evidence leaves the existing conservative catalog estimate in place.
    return null;
  }
}
