import { describe, expect, it } from "vitest";
import {
  CANDLE_HEADROOM_GB,
  DISK_TO_RESIDENT_MULTIPLIER,
  MEMORY_HEADROOM_FRACTION,
  TRANSIENT_HEADROOM_BYTES,
  blanketFloorGb,
  cheapestDeclaredTierPeakGb,
  declaredFloorHostGb,
  declaredTiers,
  hostGbForPeakGb,
  installedFloorHostGb,
  installedTierPeakGb,
  laneEvidenceEstimated,
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
  it("converts a raw peak into the smallest host that satisfies the MLX fit criterion", () => {
    // lens_turbo q4: 30.50 GiB measured. 31 GB does NOT fit it (31 × 0.9 = 27.9), 34 does.
    const peak = 32749818036 / GB;
    expect(hostGbForPeakGb(peak)).toBe(34);
    expect(hostGbForPeakGb(peak, "mlx")).toBe(34);
    expect(peak).toBeLessThanOrEqual(34 * MEMORY_HEADROOM_FRACTION);
    expect(peak).toBeGreaterThan(33 * MEMORY_HEADROOM_FRACTION);
    // The pre-review behavior, kept explicit so the regression is named: ceil(peak) UNDER-states.
    expect(Math.ceil(peak)).toBe(31);
    expect(peak).toBeGreaterThan(31 * MEMORY_HEADROOM_FRACTION);
  });

  it("uses the CANDLE lane's additive criterion on candle, not the MLX fraction", () => {
    // MAJOR 3. `vram_gate.rs` admits a candle load at `candle.vramGbByTier[tier] + HEADROOM_GB` (2.0);
    // it explicitly does NOT apply the unified-memory fraction. Dividing instead over-states against the
    // gate that actually rejects the load.
    expect(CANDLE_HEADROOM_GB).toBe(2);
    // flux2_dev bf16, the worst shipped case: 143 under the fractional form, 130 under the gate's own.
    expect(hostGbForPeakGb(128, "candle")).toBe(130);
    expect(hostGbForPeakGb(128, "mlx")).toBe(143);
    // qwen_image bf16 and its lightning edit sibling.
    expect(hostGbForPeakGb(82.5, "candle")).toBe(85);
    expect(hostGbForPeakGb(87.4, "candle")).toBe(90);
    // Tight: one GB below never clears the gate's budget.
    for (const peak of [128, 82.5, 87.4, 34.7]) {
      const shown = hostGbForPeakGb(peak, "candle");
      expect(shown).toBeGreaterThanOrEqual(peak + CANDLE_HEADROOM_GB);
      expect(shown - 1).toBeLessThan(peak + CANDLE_HEADROOM_GB);
    }
  });

  it("returns null for a missing or nonsensical peak rather than 0 or NaN", () => {
    for (const backend of ["mlx", "candle"]) {
      expect(hostGbForPeakGb(null, backend)).toBeNull();
      expect(hostGbForPeakGb(undefined, backend)).toBeNull();
      expect(hostGbForPeakGb(0, backend)).toBeNull();
      expect(hostGbForPeakGb(-5, backend)).toBeNull();
      expect(hostGbForPeakGb(Number.NaN, backend)).toBeNull();
    }
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

  it("reads the CANDLE-ONLY tier rows, which are real selectable tiers", () => {
    // krea_2_turbo's shape. `int8-convrot` has no `mlxQuantize` (it selects via `advanced.convRot`), so
    // keying this map off `tierQuantize` dropped its measured row while `installedTiers` still returned
    // the tier — collapsing the whole measured ceiling to null and falling back to the blanket. The map
    // is keyed off `isSelectableTier` for exactly that reason.
    const model = {
      candle: {
        minMemoryGb: 32,
        vramGbByTier: { q4: 25.7, q8: 35.2, "int8-convrot": 34.7 },
        measured: true,
      },
      hasVariantMatrix: true,
      variants: [{ variant: "int8-convrot", installState: "installed", footprint: null }],
    };
    expect(installedTierPeakGb(model, { backend: "candle" })).toBe(34.7);
    expect(installedFloorHostGb(model, { backend: "candle" })).toBe(37);
    // ...and the cheapest declared floor still sees the bits tiers alongside it.
    expect(cheapestDeclaredTierPeakGb(model, { backend: "candle" })).toBe(25.7);
  });

  it("drops a candle-only tier this host cannot serve, from BOTH the ceiling and the floor", () => {
    // sc-9300 / sc-11042: the tier is hidden when no live worker advertises the capability, so it must
    // not set a memory number the user can never act on either.
    const model = {
      candle: { minMemoryGb: 32, vramGbByTier: { "int8-convrot": 34.7, nvfp4: 31.5 }, measured: true },
      hasVariantMatrix: true,
      variants: [
        { variant: "int8-convrot", installState: "installed", footprint: null },
        { variant: "nvfp4", installState: "installed", footprint: null },
      ],
    };
    const ineligible = { backend: "candle", convRotEligible: false, nvfp4Eligible: false };
    expect(installedTierPeakGb(model, ineligible)).toBeNull();
    expect(cheapestDeclaredTierPeakGb(model, ineligible)).toBeNull();
    // With no eligible installed tier there is no per-tier answer at all — the blanket stands.
    expect(installedFloorHostGb(model, ineligible)).toBeNull();
    // Eligible host: both rows count, ceiling is the larger.
    expect(installedTierPeakGb(model, { backend: "candle" })).toBe(34.7);
  });

  it("degrades a candle-only tier with NO row to the q8 row, never to minMemoryGb (sc-11042)", () => {
    // `vram_gate::predicted_peak_gb`'s documented rule: minMemoryGb is the DEFAULT (lightest) tier's peak,
    // so landing there UNDER-predicts. The q8 row over-predicts, which is safe.
    const model = {
      candle: { minMemoryGb: 32, vramGbByTier: { q4: 25.7, q8: 35.2 }, measured: true },
      hasVariantMatrix: true,
      variants: [{ variant: "nvfp4", installState: "installed", footprint: null }],
    };
    expect(installedTierPeakGb(model, { backend: "candle" })).toBe(35.2);
    // NOT the blanket 32, which would be the permissive answer.
    expect(installedFloorHostGb(model, { backend: "candle" })).toBe(38);

    // No q8 row either ⇒ nothing to degrade to, so the blanket stands (mirroring the Rust's last resort).
    const noQ8 = { ...model, candle: { ...model.candle, vramGbByTier: { q4: 25.7 } } };
    expect(installedTierPeakGb(noQ8, { backend: "candle" })).toBeNull();
    expect(installedFloorHostGb(noQ8, { backend: "candle" })).toBe(32);
  });
});

describe("per-tier memory floor: the blanket is a FLOOR, not a ceiling", () => {
  // BLOCKER 1. `candle.minMemoryGb` is "the measured overall-peak of the DEFAULT (lightest, typically q4)
  // hosted tier, plus headroom ... heavier hosted tiers (q8 / bf16) can exceed this value" (schema), and
  // `vram_gate.rs:180` says landing on it is "the WRONG floor". So an install set with ONE unmeasured tier
  // must not report the blanket alone.
  function fluxDev() {
    return {
      id: "flux_dev",
      candle: { minMemoryGb: 24, vramGbByTier: { q4: 21.3, q8: 31.8 }, measured: true },
      hasVariantMatrix: true,
      variants: [
        { variant: "q4", installState: "installed", footprint: null },
        { variant: "q8", installState: "installed", footprint: null },
        { variant: "bf16", installState: "installed", footprint: null },
      ],
    };
  }

  it("takes the max of the blanket and the measured tiers when coverage is incomplete", () => {
    const model = fluxDev();
    // bf16 has no row, so the strict ceiling is unknown...
    expect(installedTierPeakGb(model, { backend: "candle" })).toBeNull();
    // ...but the measured q8 of 31.8 still needs a 34 GB host, which is ABOVE the blanket 24.
    expect(installedFloorHostGb(model, { backend: "candle" })).toBe(34);
    expect(blanketFloorGb(model, "candle")).toBe(24);
  });

  it("still reports the blanket when it is the HIGHER of the two", () => {
    // sd3_5_medium's shape: q8 unrowed, and the blanket 36 already covers the measured bf16 of 33
    // (33 + 2 = 35). The max must not lower the curated number either.
    const model = {
      candle: { minMemoryGb: 36, vramGbByTier: { q4: 28.5, bf16: 33 }, measured: false },
      hasVariantMatrix: true,
      variants: [
        { variant: "q4", installState: "installed", footprint: null },
        { variant: "q8", installState: "installed", footprint: null },
        { variant: "bf16", installState: "installed", footprint: null },
      ],
    };
    expect(installedFloorHostGb(model, { backend: "candle" })).toBe(36);
  });

  it("reports the per-tier figure ALONE when every installed tier is measured", () => {
    // The point of the story: no blanket max when the evidence is complete, or krea_realtime_14b's q4
    // user would still be told 64.
    const model = fluxDev();
    model.variants = model.variants.filter((v) => v.variant !== "bf16");
    expect(installedFloorHostGb(model, { backend: "candle" })).toBe(34);
    model.variants = model.variants.filter((v) => v.variant === "q4");
    // ceil(21.3 + 2) = 24 here, which coincides with the blanket — so use q4 alone on a model whose
    // blanket is higher to prove the figure is not being floored.
    expect(installedFloorHostGb(model, { backend: "candle" })).toBe(24);
    const heavierBlanket = {
      candle: { minMemoryGb: 48, vramGbByTier: { q4: 21.3 }, measured: true },
      hasVariantMatrix: true,
      variants: [{ variant: "q4", installState: "installed", footprint: null }],
    };
    expect(installedFloorHostGb(heavierBlanket, { backend: "candle" })).toBe(24);
  });
});

describe("per-tier memory floor: an ESTIMATED candle lane is floored at its blanket", () => {
  // MAJOR 4. `candle.measured === false` means the per-tier rows AND the blanket are both estimates
  // (the flag covers them together), so neither may lower the other.
  function lensTurbo(measured) {
    return {
      candle: { minMemoryGb: 44, vramGbByTier: { q4: 37.3, q8: 42, bf16: 52 }, measured },
      hasVariantMatrix: true,
      variants: [{ variant: "q4", installState: "installed", footprint: null }],
    };
  }

  it("takes the max when the lane says ESTIMATED", () => {
    expect(laneEvidenceEstimated(lensTurbo(false), "candle")).toBe(true);
    // ceil(37.3 + 2) = 40, floored at the curated 44.
    expect(installedFloorHostGb(lensTurbo(false), { backend: "candle" })).toBe(44);
    expect(declaredFloorHostGb(lensTurbo(false), { backend: "candle" })).toBe(44);
  });

  it("takes the per-tier figure alone when the lane says MEASURED", () => {
    expect(laneEvidenceEstimated(lensTurbo(true), "candle")).toBe(false);
    expect(installedFloorHostGb(lensTurbo(true), { backend: "candle" })).toBe(40);
  });

  it("has no MLX counterpart: the footprint harness measures by construction", () => {
    // There is no `mlx.measured` field, so the flag must never be consulted on the MLX lane — otherwise a
    // model that declares `candle.measured: false` would have its MLX answer floored by an MLX blanket.
    const model = {
      mlx: { minMemoryGb: 60 },
      candle: { minMemoryGb: 44, measured: false },
      hasVariantMatrix: true,
      variants: [
        { variant: "q4", installState: "installed", footprint: { peakMemoryBytes: 32749818036 } },
      ],
    };
    expect(laneEvidenceEstimated(model, "mlx")).toBe(false);
    // ceil(30.50 / 0.9) = 34, NOT floored at the MLX blanket 60.
    expect(installedFloorHostGb(model, { backend: "mlx" })).toBe(34);
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
