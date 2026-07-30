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

// sc-15036: a fine-tuned Mage-Flow base is the case the "no declared floor ⇒ unchanged" escape
// hatch (SCOPE note 3) was NOT written for. The builtin `mage_flow_base` declares no
// `mlx.minMemoryGb` but defaults to its pre-quantized q4 tier; a fine-tune is DENSE bf16 and
// deliberately declares no `mlx.quantize`, so it carries ~2.4x the resident weights while
// advertising the same 2048² ladder. Without a floor it would offer 2048² on any Mac — and an MLX
// overcommit is an uncatchable SIGKILL, not a recoverable error.
//
// Discriminating: the SAME entry and the SAME host, with and without the floor.
describe("a dense fine-tuned Mage base anchors the >1536 gate", () => {
  const ladder = ["1024x1024", "1536x1536", "2048x1024", "2048x2048"];
  const withFloor = { mlx: { minMemoryGb: 20 } };
  const noFloor = { mlx: {} };

  it("withholds 2048x2048 on a host that cannot hold it, and only that bucket", () => {
    // 20 + 13 x ((2048^2 - 1536^2) / 1e6) = 43.9 GB, needing ~48.7 GB at the 0.9 headroom fraction.
    expect(fitsResolutionOptions(withFloor, ladder, 32, { backend: "mlx" })).toEqual([
      "1024x1024",
      "1536x1536",
      "2048x1024",
    ]);
    // 2048x1024 is 2.10 MP — BELOW the 2.36 MP baseline — so it is always offered, not gated.
    expect(fitsResolutionOptions(withFloor, ladder, 8, { backend: "mlx" })).toEqual([
      "1024x1024",
      "1536x1536",
      "2048x1024",
    ]);
    // A 64 GB Mac clears it.
    expect(fitsResolutionOptions(withFloor, ladder, 64, { backend: "mlx" })).toEqual(ladder);
  });

  it("would offer 2048x2048 on the same 32 GB host with no floor declared", () => {
    // The pre-fix behavior, kept here so the assertion above is visibly about the FLOOR and not
    // about the ladder or the host.
    expect(fitsResolutionOptions(noFloor, ladder, 32, { backend: "mlx" })).toEqual(ladder);
  });
});

// sc-15400: the gate used the blanket `mlx.minMemoryGb` — ONE per-model integer that has to admit the
// model's heaviest INSTALLABLE tier — so it was TIER-BLIND: a host that only installed q4 was measured
// against the bf16 worst case and had high-res buckets hidden that its installed tier renders fine.
// `floorGb` now prefers the tier's MEASURED `footprint.peakMemoryBytes`.
//
// The three byte values are the REAL measured ladder from `builtin.models.jsonc` (krea_realtime_14b, the
// only shipped entry with all three tiers measured): q4 29,957,689,344 = 27.90 GiB, q8 36,980,832,256 =
// 34.44 GiB, bf16 50,154,536,960 = 46.71 GiB. They are borrowed here as MEASUREMENTS only — the model
// under test is image-shaped (a >1536² Krea 2 ladder), which is the lane this gate actually serves.
describe("per-tier measured footprint beats the blanket floor", () => {
  const KREA_Q4_BYTES = 29957689344; // 27.90 GiB
  const KREA_Q8_BYTES = 36980832256; // 34.44 GiB
  const KREA_BF16_BYTES = 50154536960; // 46.71 GiB

  // A tiered image model: blanket floor 64, plus a real per-tier ladder. `installState` per variant is
  // what `installedTiers` reads.
  function tiered({ installed = ["q4"], peaks = {} } = {}) {
    const bytes = { q4: KREA_Q4_BYTES, q8: KREA_Q8_BYTES, bf16: KREA_BF16_BYTES, ...peaks };
    return kreaModel({
      mlx: { minMemoryGb: 64 },
      candle: undefined,
      hasVariantMatrix: true,
      variants: ["q4", "q8", "bf16"].map((variant) => ({
        variant,
        installState: installed.includes(variant) ? "installed" : "missing",
        footprint:
          bytes[variant] === null
            ? { diskSizeBytes: 1 }
            : { diskSizeBytes: 1, peakMemoryBytes: bytes[variant] },
      })),
    });
  }

  // Required at 2048² = floor + 13 x 1.835008 MP = floor + 23.855; the host must clear that / 0.9.
  // q4 needs a 57.51 GB host, q8 64.77, bf16 78.41, and the blanket 64 needs 97.62.
  it("offers 2048^2 at q4 and withholds it at bf16 on the SAME 64 GB host", () => {
    const model = tiered({ installed: ["q4", "q8", "bf16"] });
    expect(resolutionFitsMemory(model, "2048x2048", 64, { backend: "mlx", tier: "q4" })).toBe(true);
    expect(resolutionFitsMemory(model, "2048x2048", 64, { backend: "mlx", tier: "q8" })).toBe(false);
    expect(resolutionFitsMemory(model, "2048x2048", 64, { backend: "mlx", tier: "bf16" })).toBe(
      false,
    );
    // A 70 GB host clears q8 too, but still not bf16 — so the tier, not the host, is doing the work.
    expect(resolutionFitsMemory(model, "2048x2048", 70, { backend: "mlx", tier: "q8" })).toBe(true);
    expect(resolutionFitsMemory(model, "2048x2048", 70, { backend: "mlx", tier: "bf16" })).toBe(
      false,
    );
  });

  it("would withhold 2048^2 at EVERY tier if it still read the blanket 64", () => {
    // The pre-fix behavior on the same host: with no per-tier footprints to prefer, the blanket floor
    // needs a 97.62 GB host, so 2048^2 is hidden even for the q4 tier that measurably fits.
    const blanketOnly = tiered({ peaks: { q4: null, q8: null, bf16: null } });
    expect(
      resolutionFitsMemory(blanketOnly, "2048x2048", 64, { backend: "mlx", tier: "q4" }),
    ).toBe(false);
    // ...and the fixed version offers it. Same model shape, same host, only the footprints differ.
    expect(resolutionFitsMemory(tiered(), "2048x2048", 64, { backend: "mlx", tier: "q4" })).toBe(
      true,
    );
  });

  it("uses the tier's measured peak as the floor verbatim, not the blanket integer", () => {
    const model = tiered({ installed: ["q4", "q8", "bf16"] });
    expect(predictedResolutionPeakGb(model, "1024x1024", "mlx", "q4")).toBeCloseTo(27.9003, 3);
    expect(predictedResolutionPeakGb(model, "1024x1024", "mlx", "q8")).toBeCloseTo(34.4411, 3);
    expect(predictedResolutionPeakGb(model, "1024x1024", "mlx", "bf16")).toBeCloseTo(46.7101, 3);
    // No tier passed and nothing installed ⇒ the blanket floor.
    expect(predictedResolutionPeakGb(tiered({ installed: [] }), "1024x1024", "mlx")).toBe(64);
  });

  it("without an explicit tier, budgets the CEILING over the installed tiers", () => {
    // Only q4 on disk ⇒ q4's 27.90 governs, and 2048^2 is offered on 64 GB.
    expect(
      resolutionFitsMemory(tiered({ installed: ["q4"] }), "2048x2048", 64, { backend: "mlx" }),
    ).toBe(true);
    // bf16 ALSO on disk ⇒ the picker can switch to it, so the 46.71 ceiling governs and it is withheld.
    // Under-stating here would offer a resolution that then OOMs.
    expect(
      resolutionFitsMemory(tiered({ installed: ["q4", "bf16"] }), "2048x2048", 64, {
        backend: "mlx",
      }),
    ).toBe(false);
  });

  it("falls back to the blanket floor when an INSTALLED tier has no measurement", () => {
    // q4 measured, q8 installed-but-unmeasured ⇒ the installed set has no known bound, so the curated
    // blanket floor (64, needing 97.62 GB) governs rather than q4's 27.90.
    const partial = tiered({ installed: ["q4", "q8"], peaks: { q8: null } });
    expect(resolutionFitsMemory(partial, "2048x2048", 64, { backend: "mlx" })).toBe(false);
    expect(predictedResolutionPeakGb(partial, "1024x1024", "mlx")).toBe(64);
  });

  it("falls back to the blanket floor when the SELECTED tier has no measurement", () => {
    const model = tiered({ peaks: { bf16: null } });
    expect(predictedResolutionPeakGb(model, "1024x1024", "mlx", "bf16")).toBe(64);
    // A tier key the model does not declare must not silently pick another tier's number either.
    expect(predictedResolutionPeakGb(model, "1024x1024", "mlx", "nvfp4")).toBe(64);
  });

  it("ignores the non-quant 'training' and single-variant 'default' pseudo-tiers", () => {
    // Neither is a generation tier; a footprint on one must never become the model's floor. Both are
    // marked installed here, so only the tier filter can keep them out.
    const model = kreaModel({
      mlx: { minMemoryGb: 48 },
      candle: undefined,
      hasVariantMatrix: true,
      variants: [
        { variant: "training", installState: "installed", footprint: { peakMemoryBytes: 1 } },
        { variant: "default", installState: "installed", footprint: { peakMemoryBytes: 1 } },
      ],
    });
    expect(predictedResolutionPeakGb(model, "1024x1024", "mlx")).toBe(48);
  });

  it("leaves an untiered model byte-identical (the blanket floor still governs)", () => {
    // Krea 2 Raw/Turbo today: a floor and >1536² buckets, but no measured per-tier footprints.
    expect(fitsResolutionOptions(kreaModel(), KREA_RES, 48, { backend: "mlx" })).toEqual(
      fitsResolutionOptions(kreaModel(), KREA_RES, 48, { backend: "mlx", tier: "q4" }),
    );
    expect(predictedResolutionPeakGb(kreaModel(), "1024x1024", "mlx", "q4")).toBe(48);
    expect(predictedResolutionPeakGb(kreaModel(), "1024x1024", "candle", "q4")).toBe(32);
  });
});
