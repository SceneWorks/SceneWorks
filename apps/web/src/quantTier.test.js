import { describe, expect, it } from "vitest";
import {
  defaultTierSelection,
  installedTiers,
  isConvRotTier,
  isSelectableTier,
  shouldShowTierPicker,
  tierLabel,
  tierQuantize,
  INT8_CONVROT_TIER,
} from "./quantTier.js";

// Build a /models-shaped model with a variant matrix. `installed` is the set of tier keys whose
// files are present (installState "installed"); every other declared tier reports "missing".
function matrixModel({ tiers = ["q4", "q8", "bf16"], installed = [], defaultTier = "q4" } = {}) {
  return {
    id: "z_image_turbo",
    hasVariantMatrix: true,
    variants: tiers.map((tier) => ({
      variant: tier,
      default: tier === defaultTier,
      installState: installed.includes(tier) ? "installed" : "missing",
    })),
  };
}

describe("quantTier mapping", () => {
  it("maps known tiers to mlxQuantize values (bf16→0, q8→8, q4→4)", () => {
    expect(tierQuantize("bf16")).toBe(0);
    expect(tierQuantize("q8")).toBe(8);
    expect(tierQuantize("q4")).toBe(4);
  });

  it("returns null for the 'default' pseudo-variant and unknown keys", () => {
    expect(tierQuantize("default")).toBe(null);
    expect(tierQuantize("q2")).toBe(null);
    expect(tierQuantize(undefined)).toBe(null);
  });

  it("labels known tiers and falls back to the raw key", () => {
    expect(tierLabel("bf16")).toBe("Full precision (bf16)");
    expect(tierLabel("q8")).toBe("Q8 (balanced)");
    expect(tierLabel("q4")).toBe("Q4 (smallest)");
    expect(tierLabel(INT8_CONVROT_TIER)).toBe("INT8-ConvRot (candle, sm_89+)");
    expect(tierLabel("mystery")).toBe("mystery");
  });
});

describe("INT8-ConvRot tier (sc-9300)", () => {
  it("is a selectable tier but has no mlxQuantize value", () => {
    expect(isConvRotTier(INT8_CONVROT_TIER)).toBe(true);
    expect(isConvRotTier("q4")).toBe(false);
    expect(isSelectableTier(INT8_CONVROT_TIER)).toBe(true);
    // It rides a distinct advanced.convRot signal, not mlxQuantize — so tierQuantize is null.
    expect(tierQuantize(INT8_CONVROT_TIER)).toBe(null);
  });

  it("is offered only when convRotEligible (candle + sm_89 worker present)", () => {
    const model = matrixModel({
      tiers: ["q4", INT8_CONVROT_TIER, "bf16"],
      installed: ["q4", INT8_CONVROT_TIER, "bf16"],
    });
    // Eligible host (default): the tier appears, ordered between q4 and bf16.
    expect(installedTiers(model)).toEqual(["q4", INT8_CONVROT_TIER, "bf16"]);
    // Ineligible host (macOS/MLX or pre-Ada NVIDIA — no int8_convrot worker): the tier is hidden.
    expect(installedTiers(model, { convRotEligible: false })).toEqual(["q4", "bf16"]);
  });

  it("is never seeded as the default selection on an ineligible host", () => {
    const model = matrixModel({
      tiers: [INT8_CONVROT_TIER, "bf16"],
      installed: [INT8_CONVROT_TIER, "bf16"],
      defaultTier: INT8_CONVROT_TIER,
    });
    // Eligible: the declared default (convrot) is picked.
    expect(defaultTierSelection(model, null)).toBe(INT8_CONVROT_TIER);
    // Ineligible: convrot is filtered out, so the remaining installed tier (bf16) is picked.
    expect(defaultTierSelection(model, null, { convRotEligible: false })).toBe("bf16");
  });

  it("does not count a hidden convrot tier toward the picker's >1 threshold", () => {
    const model = matrixModel({
      tiers: ["bf16", INT8_CONVROT_TIER],
      installed: ["bf16", INT8_CONVROT_TIER],
    });
    expect(shouldShowTierPicker(model)).toBe(true);
    // On an ineligible host only bf16 remains → single tier → no picker.
    expect(shouldShowTierPicker(model, { convRotEligible: false })).toBe(false);
  });
});

describe("installedTiers", () => {
  it("returns only installed quant tiers, in smallest→largest order", () => {
    const model = matrixModel({ installed: ["bf16", "q4"] });
    expect(installedTiers(model)).toEqual(["q4", "bf16"]);
  });

  it("returns [] for a model with no variant matrix", () => {
    expect(installedTiers({ id: "boogu", hasVariantMatrix: false, variants: [] })).toEqual([]);
    expect(installedTiers({ id: "x" })).toEqual([]);
    expect(installedTiers(undefined)).toEqual([]);
  });

  it("excludes the single-variant 'default' pseudo-tier", () => {
    const single = {
      id: "single",
      hasVariantMatrix: false,
      variants: [{ variant: "default", installState: "installed", default: true }],
    };
    expect(installedTiers(single)).toEqual([]);
  });

  it("excludes tiers that are declared but not installed", () => {
    const model = matrixModel({ installed: ["q4"] });
    expect(installedTiers(model)).toEqual(["q4"]);
  });
});

describe("shouldShowTierPicker", () => {
  it("shows the picker only when more than one tier is installed", () => {
    expect(shouldShowTierPicker(matrixModel({ installed: ["q4", "bf16"] }))).toBe(true);
    expect(shouldShowTierPicker(matrixModel({ installed: ["q4"] }))).toBe(false);
    expect(shouldShowTierPicker(matrixModel({ installed: [] }))).toBe(false);
    expect(shouldShowTierPicker({ id: "x", hasVariantMatrix: false })).toBe(false);
  });
});

describe("defaultTierSelection", () => {
  it("prefers the last-used tier when it is still installed", () => {
    const model = matrixModel({ installed: ["q4", "q8", "bf16"] });
    expect(defaultTierSelection(model, "q8")).toBe("q8");
    expect(defaultTierSelection(model, "bf16")).toBe("bf16");
  });

  it("ignores a last-used tier that is no longer installed", () => {
    const model = matrixModel({ installed: ["q4", "bf16"] });
    // q8 was last used but is now uninstalled → fall through to the declared default (q4).
    expect(defaultTierSelection(model, "q8")).toBe("q4");
  });

  it("falls back to the declared default tier when installed", () => {
    const model = matrixModel({ installed: ["q8", "bf16"], defaultTier: "q8" });
    expect(defaultTierSelection(model, null)).toBe("q8");
  });

  it("picks an installed declared default when there is no last-used tier", () => {
    const model = matrixModel({ installed: ["q4", "bf16"], defaultTier: "q4" });
    // Declared default q4 is installed → picked (step 2, ahead of the base default).
    expect(defaultTierSelection(model, undefined)).toBe("q4");
  });

  it("uses q8 as the base default when no last-used or declared default applies (epic 10721)", () => {
    // No last-used, no installed declared default → the base generation default is q8, not the
    // washed q4 (was q4 before epic 10721). Clamped to installed.
    const model = matrixModel({ installed: ["q4", "q8", "bf16"], defaultTier: "none" });
    expect(defaultTierSelection(model, undefined)).toBe("q8");
  });

  it("clamps the q8 base default to the smallest installed tier when q8 is absent", () => {
    // Only q4 (+bf16) on disk, no last-used / declared default → falls to the smallest installed
    // tier (q4), never forcing a heavier tier the machine didn't install.
    const model = matrixModel({ installed: ["q4", "bf16"], defaultTier: "none" });
    expect(defaultTierSelection(model, undefined)).toBe("q4");
  });

  it("falls back to the first installed tier when neither default nor q8 is present", () => {
    const model = matrixModel({ tiers: ["q4", "bf16"], installed: ["q4", "bf16"], defaultTier: "none" });
    expect(defaultTierSelection(model, undefined)).toBe("q4");
  });

  it("returns null when nothing is installed", () => {
    expect(defaultTierSelection(matrixModel({ installed: [] }), null)).toBe(null);
    expect(defaultTierSelection({ id: "x", hasVariantMatrix: false }, null)).toBe(null);
  });
});
