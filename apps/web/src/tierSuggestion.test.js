import { describe, expect, it } from "vitest";
import {
  DISK_TO_RESIDENT_MULTIPLIER,
  MEMORY_HEADROOM_FRACTION,
  TRANSIENT_HEADROOM_BYTES,
  blanketFloorGb,
  cheapestDeclaredTierPeakGb,
  declaredTiers,
  hostGbForPeakGb,
  installedTierPeakGb,
  suggestTier,
  tierFits,
  variantFootprintBytes,
} from "./tierSuggestion.js";

const GB = 1024 * 1024 * 1024;

// Build a /models-shaped quant-matrix model. Each entry in `tiers` may be a bare tier key (which
// gets a disk-only footprint sized from `diskGb`) or an object { variant, diskGb, residentGb } to
// exercise the measured-vs-estimate paths. Defaults roughly mirror a real z-image-class model:
// q4 ~4 GB on disk, q8 ~8 GB, bf16 ~16 GB.
function matrixModel(tiers = defaultTiers()) {
  return {
    id: "z_image_turbo",
    hasVariantMatrix: true,
    variants: tiers.map((tier) => {
      const spec = typeof tier === "string" ? { variant: tier } : tier;
      const footprint = {};
      if (spec.diskGb != null) {
        footprint.diskSizeBytes = spec.diskGb * GB;
      }
      if (spec.residentGb != null) {
        footprint.residentMemoryBytes = spec.residentGb * GB;
      }
      if (spec.peakGb != null) {
        footprint.peakMemoryBytes = spec.peakGb * GB;
      }
      return {
        variant: spec.variant,
        installState: spec.installed ? "installed" : "missing",
        downloadSizeBytes: spec.downloadGb != null ? spec.downloadGb * GB : null,
        footprint: Object.keys(footprint).length ? footprint : null,
      };
    }),
  };
}

function defaultTiers() {
  return [
    { variant: "q4", diskGb: 4 },
    { variant: "q8", diskGb: 8 },
    { variant: "bf16", diskGb: 16 },
  ];
}

describe("variantFootprintBytes", () => {
  it("uses measured peakMemoryBytes verbatim when present (the true ceiling)", () => {
    // sc-8516: peak is the memory a generation must fit, so it wins over resident when measured.
    const result = variantFootprintBytes({
      variant: "bf16",
      footprint: {
        diskSizeBytes: 16 * GB,
        residentMemoryBytes: 20 * GB,
        peakMemoryBytes: 34 * GB,
      },
    });
    expect(result).toEqual({ bytes: 34 * GB, measured: true });
  });

  it("uses measured residentMemoryBytes + fixed transient when peak is absent", () => {
    const result = variantFootprintBytes({
      variant: "bf16",
      footprint: { diskSizeBytes: 16 * GB, residentMemoryBytes: 20 * GB },
    });
    expect(result).toEqual({ bytes: 20 * GB + TRANSIENT_HEADROOM_BYTES, measured: true });
  });

  it("estimates diskSizeBytes × multiplier + transient when memory is not measured", () => {
    const result = variantFootprintBytes({ variant: "q4", footprint: { diskSizeBytes: 4 * GB } });
    expect(result.measured).toBe(false);
    expect(result.bytes).toBe(
      Math.round(4 * GB * DISK_TO_RESIDENT_MULTIPLIER) + TRANSIENT_HEADROOM_BYTES,
    );
  });

  it("falls back to downloadSizeBytes × multiplier + transient when no footprint object", () => {
    const result = variantFootprintBytes({ variant: "q4", downloadSizeBytes: 4 * GB, footprint: null });
    expect(result.measured).toBe(false);
    expect(result.bytes).toBe(
      Math.round(4 * GB * DISK_TO_RESIDENT_MULTIPLIER) + TRANSIENT_HEADROOM_BYTES,
    );
  });

  it("returns null when nothing is estimable", () => {
    expect(variantFootprintBytes({ variant: "q4", footprint: null })).toBe(null);
    expect(variantFootprintBytes({ variant: "q4" })).toBe(null);
    expect(variantFootprintBytes(undefined)).toBe(null);
  });
});

describe("declaredTiers", () => {
  it("returns real quant tiers highest-fidelity first, regardless of install state", () => {
    expect(declaredTiers(matrixModel())).toEqual(["bf16", "q8", "q4"]);
  });

  it("excludes the single-variant default pseudo-tier and unknown keys", () => {
    const model = {
      hasVariantMatrix: true,
      variants: [{ variant: "default" }, { variant: "q4" }, { variant: "mystery" }],
    };
    expect(declaredTiers(model)).toEqual(["q4"]);
  });

  it("returns [] for a non-matrix model", () => {
    expect(declaredTiers({ hasVariantMatrix: false, variants: [] })).toEqual([]);
    expect(declaredTiers(undefined)).toEqual([]);
  });
});

describe("tierFits", () => {
  it("treats an unknown memory signal as fitting (never withhold)", () => {
    const bf16 = { variant: "bf16", footprint: { diskSizeBytes: 16 * GB } };
    expect(tierFits(bf16, null)).toBe(true);
    expect(tierFits(bf16, undefined)).toBe(true);
  });

  it("treats an unestimable footprint as fitting", () => {
    expect(tierFits({ variant: "q4", footprint: null }, 8)).toBe(true);
  });

  it("fits when the estimated peak is under the headroom budget", () => {
    // q4: 4 GB disk × 1.0 + 14 GB transient = 18 GB peak. 32 GB × 0.9 = 28.8 GB budget → fits.
    const q4 = { variant: "q4", footprint: { diskSizeBytes: 4 * GB } };
    expect(tierFits(q4, 32)).toBe(true);
  });

  it("does not fit when the estimated peak exceeds the headroom budget", () => {
    // bf16: 16 GB disk × 1.0 + 14 GB transient = 30 GB peak. On a 24 GB host budget is 24 × 0.9 =
    // 21.6 GB → no.
    const bf16 = { variant: "bf16", footprint: { diskSizeBytes: 16 * GB } };
    expect(tierFits(bf16, 24)).toBe(false);
  });

  it("respects the exact headroom boundary", () => {
    // A peak exactly at budget fits; a byte over does not. Solve for the disk size whose estimated
    // peak (disk × MULT + transient) lands exactly on the budget.
    const budgetGb = 30;
    const budgetBytes = budgetGb * GB * MEMORY_HEADROOM_FRACTION; // 27 GB peak we target
    const diskBytes = (budgetBytes - TRANSIENT_HEADROOM_BYTES) / DISK_TO_RESIDENT_MULTIPLIER;
    const atBudget = { variant: "q8", footprint: { diskSizeBytes: diskBytes } };
    const overBudget = { variant: "q8", footprint: { diskSizeBytes: diskBytes + GB } };
    expect(tierFits(atBudget, budgetGb)).toBe(true);
    expect(tierFits(overBudget, budgetGb)).toBe(false);
  });
});

describe("suggestTier", () => {
  it("suggests q4 on a 32 GB host when the larger tiers overflow the budget (acceptance)", () => {
    // Peak-based budget on 32 GB = 32 × 0.9 = 28.8 GB. Estimated peak = diskGb × 1.0 + 14 GB transient.
    // Size q8/bf16 to exceed the budget so only q4 fits (a 32 GB user sees q4 pre-selected).
    const model = matrixModel([
      { variant: "q4", diskGb: 4 }, // 4 + 14 = 18 GB peak < 28.8 → fits
      { variant: "q8", diskGb: 18 }, // 18 + 14 = 32 GB peak > 28.8 → no
      { variant: "bf16", diskGb: 24 }, // 24 + 14 = 38 GB peak > 28.8 → no
    ]);
    expect(suggestTier(model, 32)).toBe("q4");
  });

  it("suggests q8 when bf16 is too big but q8 fits on a 32 GB host", () => {
    // Budget 28.8 GB (32 × 0.9); estimated peak = diskGb + 14 GB transient.
    const model = matrixModel([
      { variant: "q4", diskGb: 4 }, // 18 GB peak
      { variant: "q8", diskGb: 10 }, // 10 + 14 = 24 GB peak < 28.8 → fits
      { variant: "bf16", diskGb: 24 }, // 38 GB peak → no
    ]);
    // q8 is higher fidelity than q4 and fits → preferred over q4.
    expect(suggestTier(model, 32)).toBe("q8");
  });

  it("suggests bf16 on a 512 GB Studio", () => {
    expect(suggestTier(matrixModel(), 512)).toBe("bf16");
  });

  it("prefers a measured peak footprint over the disk estimate", () => {
    // bf16 disk is small (would estimate as fitting) but MEASURED peak is huge → excluded on 32 GB
    // (budget 28.8 GB). Exercises the measured-footprint path the way real manifest data flows.
    const model = matrixModel([
      { variant: "q4", diskGb: 4 },
      { variant: "bf16", diskGb: 8, peakGb: 40 }, // measured peak 40 GB > 28.8 → no
    ]);
    expect(suggestTier(model, 32)).toBe("q4");
  });

  it("suggests a MEASURED tier whose peak fits on a right-sized host (sc-8516 calibration)", () => {
    // Real measured lens_turbo-class numbers: q4 peak ≈ 30.5 GB. On a 48 GB Mac (budget 43.2 GB) it
    // fits; on a 32 GB Mac (budget 28.8 GB) it does not — the exact threshold behavior the harness
    // measured. Only q4 is declared here (the only lens tier sc-8516 measured).
    const lensQ4 = matrixModel([{ variant: "q4", diskGb: 20, peakGb: 30.5 }]);
    expect(suggestTier(lensQ4, 48)).toBe("q4");
    expect(tierFits(lensQ4.variants[0], 32)).toBe(false);
    expect(tierFits(lensQ4.variants[0], 48)).toBe(true);
  });

  it("Wan A14B (sc-10042): measured q4 peak fits every Mac; heavier tiers warn only on small hosts", () => {
    // Real manifest data: sc-10049 MEASURED the q4 video peak at ~24.5 GiB on-device; q8/bf16 are
    // disk-estimated. A14B is MoE — only ONE 14B expert is resident at a time — so q4's measured peak
    // sits far below its ~26.7 GiB on-disk size, which is exactly why the old blanket model-level
    // minMemoryGb 133 (both experts dense) grossly over-warned q4/q8-default users.
    const wan = matrixModel([
      { variant: "q4", diskGb: 26.7, peakGb: 24.5 }, // MEASURED peak (not the disk estimate)
      { variant: "q8", diskGb: 39.8 }, // est 39.8 + 14 = 53.8 GB
      { variant: "bf16", diskGb: 64.3 }, // est 64.3 + 14 = 78.3 GB
    ]);
    const [q4, q8, bf16] = wan.variants;
    // 128 GB Mac (budget 115.2 GB): every tier fits — matches on-device reality (even bf16 A14B ran).
    expect(tierFits(q4, 128)).toBe(true);
    expect(tierFits(q8, 128)).toBe(true);
    expect(tierFits(bf16, 128)).toBe(true);
    expect(suggestTier(wan, 128)).toBe("bf16");
    // 32 GB Mac (budget 28.8 GB): only the measured q4 fits; the heavier tiers are correctly flagged
    // (this is the per-tier "may exceed memory" the UI now shows, replacing the blanket 133 warning).
    expect(tierFits(q4, 32)).toBe(true);
    expect(tierFits(q8, 32)).toBe(false);
    expect(tierFits(bf16, 32)).toBe(false);
    expect(suggestTier(wan, 32)).toBe("q4");
  });

  it("falls back to the smallest tier when nothing fits", () => {
    const model = matrixModel([
      { variant: "q4", diskGb: 40 }, // 40 + 14 = 54 GB peak est
      { variant: "bf16", diskGb: 80 },
    ]);
    // Tiny 8 GB host, every tier over budget → smallest declared tier (q4).
    expect(suggestTier(model, 8)).toBe("q4");
  });

  it("suggests the highest-fidelity tier when memory is unknown", () => {
    expect(suggestTier(matrixModel(), null)).toBe("bf16");
  });

  it("returns null for a non-matrix model", () => {
    expect(suggestTier({ hasVariantMatrix: false, variants: [] }, 32)).toBe(null);
  });
});

describe("suggestTier — real SANA-Sprint footprints (epic 10721 Auto default)", () => {
  // Actual on-disk tier sizes for SceneWorks/Sana_Sprint_1.6B_1024px_mlx (from the live catalog).
  const sanaSprint = () =>
    matrixModel([
      { variant: "q4", diskGb: 5.19 },
      { variant: "q8", diskGb: 6.54 },
      { variant: "bf16", diskGb: 9.05 },
    ]);

  it("suggests bf16 for a small model on a roomy Mac (Michael's 128 GB case)", () => {
    // bf16 peak ≈ 9.05 + 14 transient = ~23 GB, well under 128 × 0.9 = 115 GB → full precision.
    expect(suggestTier(sanaSprint(), 128)).toBe("bf16");
  });

  it("still leans as high as fits on a mid Mac, and steps down on a tiny one", () => {
    // 64 GB: bf16 ~23 GB < 57.6 budget → bf16.
    expect(suggestTier(sanaSprint(), 64)).toBe("bf16");
    // 16 GB: every tier's peak (~19–23 GB) exceeds the 14.4 GB budget → smallest declared (q4).
    expect(suggestTier(sanaSprint(), 16)).toBe("q4");
  });
});

// ---------------------------------------------------------------------------------------------------
// Per-tier, PER-LANE measured floors (sc-15400 + its review). The catalog-driven counterpart lives in
// memoryFloorCatalogParity.test.js, which proves the lane independence against the real manifest. These
// are the shapes the real catalog cannot currently produce — duplicate tier keys (three upstream filters
// remove them today) and the exact headroom arithmetic.
// ---------------------------------------------------------------------------------------------------
describe("per-tier memory floor: headroom conversion", () => {
  it("converts a raw peak into the smallest host that satisfies the fit criterion", () => {
    // lens_turbo q4: 30.50 GiB measured. 31 GB does NOT fit it (31 × 0.9 = 27.9), 34 does.
    const peak = 32749818036 / GB;
    expect(hostGbForPeakGb(peak)).toBe(34);
    expect(peak).toBeLessThanOrEqual(34 * MEMORY_HEADROOM_FRACTION);
    expect(peak).toBeGreaterThan(33 * MEMORY_HEADROOM_FRACTION);
    // The pre-review behavior, kept explicit so the regression is named: ceil(peak) UNDER-states.
    expect(Math.ceil(peak)).toBe(31);
    expect(peak).toBeGreaterThan(31 * MEMORY_HEADROOM_FRACTION);
  });

  it("returns null for a missing or nonsensical peak rather than 0 or NaN", () => {
    expect(hostGbForPeakGb(null)).toBeNull();
    expect(hostGbForPeakGb(undefined)).toBeNull();
    expect(hostGbForPeakGb(0)).toBeNull();
    expect(hostGbForPeakGb(-5)).toBeNull();
    expect(hostGbForPeakGb(Number.NaN)).toBeNull();
  });
});

describe("per-tier memory floor: the lane selects the evidence", () => {
  // A model carrying BOTH lanes' evidence for the same tier, as lens_turbo really does.
  function bothLanes() {
    return {
      id: "lens_turbo",
      mlx: { minMemoryGb: 60 },
      candle: { minMemoryGb: 44, vramGbByTier: { q4: 37.3, q8: 42.0, bf16: 52.0 } },
      hasVariantMatrix: true,
      variants: [
        { variant: "q4", installState: "installed", footprint: { peakMemoryBytes: 32749818036 } },
        { variant: "q8", installState: "missing", footprint: { diskSizeBytes: 1 } },
      ],
    };
  }

  it("reads the MLX footprint on mlx and candle.vramGbByTier on candle", () => {
    const model = bothLanes();
    expect(installedTierPeakGb(model, { backend: "mlx" })).toBeCloseTo(30.5006, 3);
    expect(installedTierPeakGb(model, { backend: "candle" })).toBe(37.3);
  });

  it("takes the blanket floor from the SAME lane", () => {
    const model = bothLanes();
    expect(blanketFloorGb(model, "mlx")).toBe(60);
    expect(blanketFloorGb(model, "candle")).toBe(44);
    // A lane that declares no block gets null — never the other lane's integer. qwen_image (mlx 50 /
    // candle 56) is why: the MLX number is not reliably the conservative one.
    expect(blanketFloorGb({ mlx: { minMemoryGb: 48 } }, "candle")).toBeNull();
    expect(blanketFloorGb({ candle: { minMemoryGb: 16 } }, "mlx")).toBeNull();
  });

  it("returns null on candle when only the MLX footprint exists", () => {
    // z_image's shape: a measured MLX q4 and no candle block at all.
    const model = {
      id: "z_image",
      mlx: { minMemoryGb: 48 },
      hasVariantMatrix: true,
      variants: [
        { variant: "q4", installState: "installed", footprint: { peakMemoryBytes: 20852069456 } },
      ],
    };
    expect(installedTierPeakGb(model, { backend: "mlx" })).toBeCloseTo(19.42, 2);
    expect(installedTierPeakGb(model, { backend: "candle" })).toBeNull();
    expect(cheapestDeclaredTierPeakGb(model, { backend: "candle" })).toBeNull();
  });

  it("ignores non-tier keys in candle.vramGbByTier", () => {
    // The manifest also stores int8-convrot rows there; only real bits-based tiers are generation tiers.
    const model = {
      candle: { vramGbByTier: { q4: 25.7, "int8-convrot": 34.7 } },
      hasVariantMatrix: true,
      variants: [{ variant: "q4", installState: "installed", footprint: null }],
    };
    expect(installedTierPeakGb(model, { backend: "candle" })).toBe(25.7);
    expect(cheapestDeclaredTierPeakGb(model, { backend: "candle" })).toBe(25.7);
  });
});

describe("per-tier memory floor: duplicate tier keys resolve to the MAX", () => {
  // The real catalog cannot emit duplicates (coRequisite + per-OS + first-wins de-dupe all remove them,
  // pinned by memoryFloorCatalogParity.test.js). These assert the client no longer DEPENDS on that.
  it("keeps the measured duplicate rather than letting a footprint-less one erase it", () => {
    // mage_flow's shape: main weights first, then component rows under the same key with no footprint.
    // A Map keyed blindly on `variant` would take the LAST and fall back to the blanket.
    const model = {
      hasVariantMatrix: true,
      mlx: { minMemoryGb: 48 },
      variants: [
        { variant: "q4", installState: "installed", footprint: { peakMemoryBytes: 20852069456 } },
        { variant: "q4", installState: "installed", footprint: null },
        { variant: "q4", installState: "installed", footprint: { diskSizeBytes: 1 } },
      ],
    };
    expect(installedTierPeakGb(model, { backend: "mlx" })).toBeCloseTo(19.42, 2);
  });

  it("takes the LARGER of two measured duplicates, never the smaller", () => {
    // Order-independent, and biased toward over-stating: under-stating is the direction that OOMs.
    const ascending = {
      hasVariantMatrix: true,
      variants: [
        { variant: "q8", installState: "installed", footprint: { peakMemoryBytes: 10 * GB } },
        { variant: "q8", installState: "installed", footprint: { peakMemoryBytes: 20 * GB } },
      ],
    };
    const descending = {
      hasVariantMatrix: true,
      variants: [
        { variant: "q8", installState: "installed", footprint: { peakMemoryBytes: 20 * GB } },
        { variant: "q8", installState: "installed", footprint: { peakMemoryBytes: 10 * GB } },
      ],
    };
    expect(installedTierPeakGb(ascending, { backend: "mlx" })).toBe(20);
    expect(installedTierPeakGb(descending, { backend: "mlx" })).toBe(20);
  });
});
