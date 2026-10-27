import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import { loadSdxlCudaMemoryProjection } from "./sdxl-cuda-memory-projection.js";

const root = pathToFileURL(`${resolve(dirname(fileURLToPath(import.meta.url)), "../..")}/`);
const read = (path) => readFileSync(new URL(path, root), "utf8");

describe("SDXL CUDA browser evidence projection", () => {
  it("projects both shipped observations without exposing capture paths or device identifiers", () => {
    const projection = loadSdxlCudaMemoryProjection();
    const config = JSON.parse(read("config/sdxl-cuda-memory-endpoints.json"));
    expect(projection).not.toBeNull();
    expect(projection.resident.peakBytes).toBe(config.pair.resident.overallAllocatorEnvelopeBytes);
    expect(projection.optimized.peakBytes).toBe(config.pair.sequential.overallAllocatorEnvelopeBytes);
    expect(projection.optimized.peakBytes).toBeLessThan(projection.resident.peakBytes);
    expect(projection.geometry).toEqual({ width: 1024, height: 1024, batch: 1, frames: 1, referenceCount: 0 });
    expect(projection.physicalEightGiBDeviceValidated).toBe(false);
    expect(JSON.stringify(projection)).not.toMatch(/E:\\|GPU-|artifactInventory|SCENEWORKS_SDXL_ROOT/);
  });

  it("keeps the browser allowance and reserve equal to native admission policy", () => {
    const projection = loadSdxlCudaMemoryProjection();
    const allowance = read("crates/sceneworks-worker/src/ladder_margin_policy.rs")
      .match(/pub const CANDLE_RECAPTURE_SPREAD: f64 = ([\d.]+);/);
    const reserve = read("crates/sceneworks-worker/src/fit_gate.rs")
      .match(/const DEDICATED_VRAM_ALLOCATOR_SLACK_GB: f64 = ([\d.]+);/);
    expect(allowance).not.toBeNull();
    expect(reserve).not.toBeNull();
    expect(projection.recaptureSpread).toBe(Number(allowance[1]));
    expect(projection.reserveGb).toBe(Number(reserve[1]));
  });

  it("falls back when retained evidence is missing or its authenticated peak is edited", () => {
    const temp = mkdtempSync(join(tmpdir(), "sdxl-ui-evidence-"));
    const url = pathToFileURL(`${temp}/`);
    try {
      expect(loadSdxlCudaMemoryProjection(url)).toBeNull();
      const config = JSON.parse(read("config/sdxl-cuda-memory-endpoints.json"));
      mkdirSync(join(temp, "config/sdxl-cuda-memory-endpoint-evidence"), { recursive: true });
      for (const receipt of [config.pair.resident.source, config.pair.sequential.source, config.pair.summary]) {
        writeFileSync(join(temp, receipt.path), read(receipt.path));
      }
      writeFileSync(join(temp, "config/sdxl-cuda-memory-endpoints.json"), JSON.stringify(config));
      expect(loadSdxlCudaMemoryProjection(url)).not.toBeNull();
      config.pair.sequential.overallAllocatorEnvelopeBytes -= 1024;
      writeFileSync(join(temp, "config/sdxl-cuda-memory-endpoints.json"), JSON.stringify(config));
      expect(loadSdxlCudaMemoryProjection(url)).toBeNull();
    } finally {
      rmSync(temp, { recursive: true, force: true });
    }
  });
});
