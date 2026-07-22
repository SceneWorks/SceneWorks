import { describe, expect, it } from "vitest";
import {
  BASELINE_PIXELS,
  HIGHRES_TRANSIENT_GB_PER_MP,
  fitsResolutionOptions,
  predictedResolutionPeakGb,
  resolutionFitsMemory,
} from "./resolutionMemory.js";
import { MEMORY_HEADROOM_FRACTION } from "./tierSuggestion.js";

// A Krea-shaped model: a real mlx/candle memory floor + the sc-13959 high-res ladder.
function kreaModel(overrides = {}) {
  return {
    id: "krea_2_raw",
    family: "krea_2",
    mlx: { minMemoryGb: 48 },
    candle: { minMemoryGb: 32 },
    limits: {
      resolutions: [
        "1024x1024",
        "1216x832",
        "1152x896",
        "1536x1536",
        "2048x1152",
        "2048x1408",
        "2048x2048",
      ],
    },
    ...overrides,
  };
}

const KREA_RES = kreaModel().limits.resolutions;

describe("resolutionFitsMemory", () => {
  it("always offers the historical <=1536^2 set, regardless of memory or backend", () => {
    for (const res of ["1024x1024", "1216x832", "1152x896", "1536x1536"]) {
      // Even an absurdly tiny budget cannot hide a <=baseline bucket.
      expect(resolutionFitsMemory(kreaModel(), res, 1, { backend: "mlx" })).toBe(true);
      expect(resolutionFitsMemory(kreaModel(), res, 1, { backend: "candle" })).toBe(true);
    }
    // 1536x1536 is exactly the baseline (not strictly above it) and stays offered.
    expect(1536 * 1536).toBe(BASELINE_PIXELS);
  });

  it("never withholds when the host memory reading is unknown", () => {
    for (const res of KREA_RES) {
      expect(resolutionFitsMemory(kreaModel(), res, null, { backend: "mlx" })).toBe(true);
      expect(resolutionFitsMemory(kreaModel(), res, undefined, { backend: "mlx" })).toBe(true);
      expect(resolutionFitsMemory(kreaModel(), res, NaN, { backend: "mlx" })).toBe(true);
    }
  });

  it("hides 2048^2 on a 48 GB Mac but offers it on a 128 GB Mac (mlx)", () => {
    // Discriminating boundary: a low budget must EXCLUDE the top bucket a high budget INCLUDES.
    expect(resolutionFitsMemory(kreaModel(), "2048x2048", 48, { backend: "mlx" })).toBe(false);
    expect(resolutionFitsMemory(kreaModel(), "2048x2048", 128, { backend: "mlx" })).toBe(true);
  });

  it("hides 2048^2 on a 48 GB CUDA card but offers it on an 80 GB card (candle)", () => {
    expect(resolutionFitsMemory(kreaModel(), "2048x2048", 48, { backend: "candle" })).toBe(false);
    expect(resolutionFitsMemory(kreaModel(), "2048x2048", 80, { backend: "candle" })).toBe(true);
  });

  it("gates a mid-high bucket independently of the top bucket", () => {
    // 2048x1152 (2.36 MP, just above baseline) fits a 64 GB Mac that still can't fit 2048^2 (4.19 MP).
    expect(resolutionFitsMemory(kreaModel(), "2048x1152", 64, { backend: "mlx" })).toBe(true);
    expect(resolutionFitsMemory(kreaModel(), "2048x2048", 64, { backend: "mlx" })).toBe(false);
  });

  it("leaves a model with no declared memory floor unchanged (e.g. SenseNova)", () => {
    // SenseNova ships 2048^2 today with no minMemoryGb — the gate must not start hiding its buckets,
    // even on a tiny host with a known memory reading.
    const sensenova = {
      id: "sensenova_u1_8b",
      limits: { resolutions: ["2048x2048", "1888x1248"] },
    };
    expect(resolutionFitsMemory(sensenova, "2048x2048", 16, { backend: "mlx" })).toBe(true);
    expect(resolutionFitsMemory(sensenova, "1888x1248", 16, { backend: "mlx" })).toBe(true);
  });

  it("budgets exactly at the shared 0.9 headroom fraction boundary", () => {
    // Pin a NON-default boundary so the test discriminates the actual comparison, not a constant.
    const model = kreaModel({ mlx: { minMemoryGb: 40 }, candle: undefined });
    const pixels = 2048 * 2048;
    const required =
      40 + HIGHRES_TRANSIENT_GB_PER_MP * ((pixels - BASELINE_PIXELS) / 1_000_000);
    const exactBudgetGb = required / MEMORY_HEADROOM_FRACTION;
    // At exactly the budget it fits; a hair under it does not.
    expect(resolutionFitsMemory(model, "2048x2048", exactBudgetGb, { backend: "mlx" })).toBe(true);
    expect(
      resolutionFitsMemory(model, "2048x2048", exactBudgetGb - 0.001, { backend: "mlx" }),
    ).toBe(false);
  });
});

describe("predictedResolutionPeakGb", () => {
  it("returns the bare floor at/below baseline and floor+transient above it", () => {
    expect(predictedResolutionPeakGb(kreaModel(), "1024x1024", "mlx")).toBe(48);
    expect(predictedResolutionPeakGb(kreaModel(), "1536x1536", "mlx")).toBe(48);
    const above = predictedResolutionPeakGb(kreaModel(), "2048x2048", "mlx");
    expect(above).toBeGreaterThan(48);
    const extraMp = (2048 * 2048 - BASELINE_PIXELS) / 1_000_000;
    expect(above).toBeCloseTo(48 + HIGHRES_TRANSIENT_GB_PER_MP * extraMp, 6);
  });

  it("reads the backend-specific floor", () => {
    expect(predictedResolutionPeakGb(kreaModel(), "1024x1024", "mlx")).toBe(48);
    expect(predictedResolutionPeakGb(kreaModel(), "1024x1024", "candle")).toBe(32);
  });

  it("is null when the model declares no floor or the resolution is malformed", () => {
    expect(predictedResolutionPeakGb({ id: "x" }, "2048x2048", "mlx")).toBeNull();
    expect(predictedResolutionPeakGb(kreaModel(), "not-a-size", "mlx")).toBeNull();
  });
});

describe("fitsResolutionOptions", () => {
  it("trims only the over-budget high-res buckets, preserving order", () => {
    const filtered = fitsResolutionOptions(kreaModel(), KREA_RES, 48, { backend: "mlx" });
    // The <=1536^2 set survives; 2048^2 is dropped on a 48 GB Mac.
    expect(filtered).toContain("1024x1024");
    expect(filtered).toContain("1536x1536");
    expect(filtered).not.toContain("2048x2048");
    // Order of the survivors matches the input order.
    expect(filtered).toEqual(KREA_RES.filter((r) => filtered.includes(r)));
  });

  it("returns the full list on a large host and on an unknown reading", () => {
    expect(fitsResolutionOptions(kreaModel(), KREA_RES, 256, { backend: "mlx" })).toEqual(KREA_RES);
    expect(fitsResolutionOptions(kreaModel(), KREA_RES, null, { backend: "mlx" })).toEqual(KREA_RES);
  });

  it("tolerates a non-array input", () => {
    expect(fitsResolutionOptions(kreaModel(), undefined, 48, { backend: "mlx" })).toEqual([]);
  });
});
