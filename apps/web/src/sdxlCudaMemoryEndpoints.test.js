import { describe, expect, it } from "vitest";
import { sdxlCudaMemoryEstimate } from "./sdxlCudaMemoryEndpoints.js";
import { tierFits, installedFloorHostGb } from "./tierSuggestion.js";
import { needsLabel } from "./simple/SimpleModelManager.jsx";

const projection = JSON.parse(import.meta.env.VITE_SDXL_CUDA_MEMORY_ENDPOINTS);
const model = {
  id: "sdxl", installState: "installed", hasVariantMatrix: true,
  variants: [{ variant: "q4", installState: "installed" }, { variant: "q8", installState: "missing" }],
  candle: { measured: false, minMemoryGb: 12, vramGbByTier: { q4: 12, q8: 16 } },
};

// CI_IMPACT: pure projected metadata and labels; no model, device, API, or renderer execution.
describe("SDXL measured CUDA endpoint catalog guidance", () => {
  it("keeps the actual two measured peaks separate from allowance and reserve", () => {
    const result = sdxlCudaMemoryEstimate(model, "q4", "candle");
    expect(result).not.toBeNull();
    expect(result.optimizedPeakGb).toBe(3562012672 / 1024 ** 3);
    expect(result.residentPeakGb).toBe(10004463616 / 1024 ** 3);
    expect(result.optimizedHostGb).toBe(Math.ceil(projection.optimized.peakBytes * 1.02) / 1024 ** 3 + 2);
    expect(result.residentHostGb).toBe(Math.ceil(projection.resident.peakBytes * 1.02) / 1024 ** 3 + 2);
    expect(result.description).toContain("optimized 3.32 GiB; resident 9.32 GiB");
    expect(result.description).toContain("optimized 5.38 GiB; resident 11.50 GiB");
    expect(result.description).toContain("1024 × 1024, one image");
    expect(result.description).toContain("no adapters");
    expect(result.description).toContain("does not validate a physical 8 GiB card");
  });

  it("fits the scoped Q4 default on 8 GiB using exact arithmetic, without repricing other tiers or lanes", () => {
    const result = sdxlCudaMemoryEstimate(model, "q4", "candle");
    const fit = (gb) => tierFits(model.variants[0], gb, { backend: "candle", model });
    expect(fit(8)).toBe(true);
    expect(fit(result.optimizedHostGb)).toBe(true);
    expect(fit(result.optimizedHostGb - 1 / 1024 ** 3)).toBe(false);
    expect(fit(5.3)).toBe(false);
    expect(tierFits(model.variants[1], 8, { backend: "candle", model })).toBe(false);
    expect(sdxlCudaMemoryEstimate(model, "q4", "mlx")).toBeNull();
    expect(sdxlCudaMemoryEstimate({ ...model, id: "realvisxl" }, "q4", "candle")).toBeNull();
    expect(model.candle.vramGbByTier.q4).toBe(12);
  });

  it("fails back to the conservative catalog when the projection or its request coordinate is invalid", () => {
    for (const patch of [null, {}, { ...projection, schemaVersion: 2 }, { ...projection, backend: "mlx" },
      { ...projection, geometry: { ...projection.geometry, width: 768 } },
      { ...projection, settings: { ...projection.settings, count: 2 } },
      { ...projection, resident: { ...projection.resident, peakBytes: NaN } },
      { ...projection, optimized: { ...projection.optimized, calibrationFingerprint: "other-v3" } },
      { ...projection, recaptureSpread: -1 }, { ...projection, reserveGb: 0 }]) {
      expect(sdxlCudaMemoryEstimate(model, "q4", "candle", patch)).toBeNull();
    }
  });

  it("labels only the Q4-only install as optimized and preserves a mixed install's aggregate floor", () => {
    expect(needsLabel(model, { backend: "candle", hostMemoryGb: 8 })).toBe("Q4 optimized ~5.4 GiB · can fit with optimizations");
    expect(needsLabel(model, { backend: "candle", hostMemoryGb: 5.3 })).toBe("Q4 optimized ~5.4 GiB");
    const mixed = { ...model, variants: model.variants.map((variant) => ({ ...variant, installState: "installed" })) };
    expect(needsLabel(mixed, { backend: "candle" })).toBe(`needs ${installedFloorHostGb(mixed, { backend: "candle" })} GB`);
    expect(needsLabel({ ...mixed, installState: "incomplete" }, { backend: "candle" }))
      .toBe(`needs ${installedFloorHostGb(mixed, { backend: "candle" })} GB`);
    expect(needsLabel({ ...model, variants: [{ variant: "q4", installState: "missing" }] }, { backend: "candle" }))
      .not.toContain("optimized");
    const pending = { ...model, installState: "missing", variants: [{ variant: "q4", installState: "pending", pendingArtifact: true }] };
    expect(sdxlCudaMemoryEstimate(pending, "q4", "candle")).toBeNull();
    expect(needsLabel(pending, { backend: "candle" })).not.toContain("optimized");
  });
});
