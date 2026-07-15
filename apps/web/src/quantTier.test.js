import { describe, expect, it } from "vitest";
import {
  DEFAULT_GENERATION_QUALITY,
  GENERATION_QUALITY_TIERS,
  defaultTierSelection,
  installedTiers,
  isBelowFloor,
  isConvRotTier,
  isNvfp4Tier,
  isSelectableTier,
  modelQualityFloor,
  shouldShowTierPicker,
  tierLabel,
  tierQuantize,
  INT8_CONVROT_TIER,
  NVFP4_TIER,
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

describe("NVFP4 tier (sc-11042, epic 11037)", () => {
  it("is a selectable tier but has no mlxQuantize value", () => {
    expect(isNvfp4Tier(NVFP4_TIER)).toBe(true);
    expect(isNvfp4Tier("q4")).toBe(false);
    expect(isSelectableTier(NVFP4_TIER)).toBe(true);
    // It rides a distinct advanced.quantTier signal, not mlxQuantize — so tierQuantize is null.
    // Critically NOT 4: `mlxQuantize: 4` names the int4-affine q4 tier, so sending it for NVFP4 would
    // alias one creative choice onto another (epic 11037 SC#5).
    expect(tierQuantize(NVFP4_TIER)).toBe(null);
    expect(tierLabel(NVFP4_TIER)).toBe("NVFP4 (candle, Blackwell sm_120+)");
  });

  it("is offered only when nvfp4Eligible (candle + sm_120 Blackwell worker present)", () => {
    const model = matrixModel({
      tiers: ["q4", NVFP4_TIER, "bf16"],
      installed: ["q4", NVFP4_TIER, "bf16"],
    });
    // Eligible host (default): the tier appears, ordered just above q4.
    expect(installedTiers(model)).toEqual(["q4", NVFP4_TIER, "bf16"]);
    // Ineligible host (macOS/MLX or pre-Blackwell NVIDIA — no nvfp4 worker): the tier is hidden even
    // though its files are installed.
    expect(installedTiers(model, { nvfp4Eligible: false })).toEqual(["q4", "bf16"]);
  });

  it("gates independently of the convrot tier", () => {
    const model = matrixModel({
      tiers: ["q4", NVFP4_TIER, INT8_CONVROT_TIER],
      installed: ["q4", NVFP4_TIER, INT8_CONVROT_TIER],
    });
    // An sm_89 Ada host clears ConvRot but NOT NVFP4 — the real hardware split this gate exists for.
    expect(installedTiers(model, { nvfp4Eligible: false })).toEqual(["q4", INT8_CONVROT_TIER]);
    // Both hidden (macOS/MLX).
    expect(installedTiers(model, { convRotEligible: false, nvfp4Eligible: false })).toEqual(["q4"]);
    // Both offered (an sm_120 candle host).
    expect(installedTiers(model)).toEqual(["q4", NVFP4_TIER, INT8_CONVROT_TIER]);
  });

  it("does not count a hidden nvfp4 tier toward the picker's >1 threshold", () => {
    const model = matrixModel({
      tiers: ["bf16", NVFP4_TIER],
      installed: ["bf16", NVFP4_TIER],
    });
    expect(shouldShowTierPicker(model)).toBe(true);
    // On an ineligible host only bf16 remains → single tier → no picker.
    expect(shouldShowTierPicker(model, { nvfp4Eligible: false })).toBe(false);
  });

  // epic 11037 SC#5 at the UI layer: NVFP4 is an explicit choice, never a default anyone lands on.
  it("is never the app-wide default quality vocabulary", () => {
    expect(GENERATION_QUALITY_TIERS).not.toContain(NVFP4_TIER);
    // …so even asking for it as the global default falls back to the q8 base rather than selecting it.
    const model = matrixModel({
      tiers: ["q4", NVFP4_TIER, "q8"],
      installed: ["q4", NVFP4_TIER, "q8"],
      defaultTier: null,
    });
    expect(defaultTierSelection(model, null, { defaultQuality: NVFP4_TIER })).toBe("q8");
  });

  it("is never seeded as the default over an installed q4/q8 (SC#5: no auto-swap of q4)", () => {
    const model = matrixModel({
      tiers: ["q4", NVFP4_TIER, "q8"],
      installed: ["q4", NVFP4_TIER, "q8"],
      defaultTier: null,
    });
    // On a fully NVFP4-eligible Blackwell host with the tier installed, the default is still q8 — the
    // presence of FP4 hardware never re-points an existing tier's default.
    expect(defaultTierSelection(model, null)).toBe("q8");
    // A q4 sticky stays q4 — it is NOT silently upgraded to NVFP4 on Blackwell (the Option B this
    // story rejected). This is the web-side twin of the worker's SC#5 regression guard.
    expect(defaultTierSelection(model, "q4")).toBe("q4");
    // NVFP4 is reachable as a sticky — i.e. only because the user explicitly picked it before.
    expect(defaultTierSelection(model, NVFP4_TIER)).toBe(NVFP4_TIER);
  });

  it("never participates in a quality-floor comparison", () => {
    // NVFP4 is a distinct numeric regime, not a rung on the bf16/q8/q4 ladder — ranking it against a
    // floor would be a category error, so it is never flagged as "below floor".
    expect(isBelowFloor(NVFP4_TIER, { minQualityTier: "q8" })).toBe(false);
    expect(isBelowFloor(NVFP4_TIER, { minQualityTier: "bf16" })).toBe(false);
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

describe("quantTier — video model reuse (sc-12165)", () => {
  const videoModel = (installed) => ({
    id: "bernini",
    type: "video",
    hasVariantMatrix: true,
    variants: ["q4", "q8", "bf16"].map((variant) => ({
      variant,
      installState: installed.includes(variant) ? "installed" : "missing",
    })),
  });

  it("derives installed MLX video tiers through the model-agnostic helpers", () => {
    const model = videoModel(["q4", "q8"]);
    expect(installedTiers(model)).toEqual(["q4", "q8"]);
    expect(shouldShowTierPicker(model)).toBe(true);
    expect(defaultTierSelection(model, null, { defaultQuality: "q4" })).toBe("q4");
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

  it("honors an installed declared default tier over the q8 base default", () => {
    const model = matrixModel({ installed: ["q4", "bf16"], defaultTier: "q4" });
    // Declared default q4 is installed → picked (the manifest's per-model default still wins).
    expect(defaultTierSelection(model, undefined)).toBe("q4");
  });

  it("prefers q8 by default when installed and no declared default/last-used applies (sc-10726)", () => {
    // No declared default, no last-used → the app-wide q8 base default is seeded when q8 is installed.
    const model = matrixModel({ installed: ["q4", "q8", "bf16"], defaultTier: "none" });
    expect(defaultTierSelection(model, undefined)).toBe("q8");
  });

  it("clamps the q8 base default to q4 when only q4 is installed (sc-10726)", () => {
    // Q8 base default falls back to q4 when q8 isn't on disk — never seeds a tier that isn't installed.
    const model = matrixModel({ tiers: ["q4", "q8", "bf16"], installed: ["q4"], defaultTier: "none" });
    expect(defaultTierSelection(model, undefined)).toBe("q4");
  });

  it("falls back to the first installed tier when neither a declared default, q8, nor q4 is present", () => {
    const model = matrixModel({ tiers: ["bf16"], installed: ["bf16"], defaultTier: "none" });
    expect(defaultTierSelection(model, undefined)).toBe("bf16");
  });

  it("returns null when nothing is installed", () => {
    expect(defaultTierSelection(matrixModel({ installed: [] }), null)).toBe(null);
    expect(defaultTierSelection({ id: "x", hasVariantMatrix: false }, null)).toBe(null);
  });
});

// Global "default generation quality" setting (epic 10721 / sc-10728): the app-wide base default is
// no longer hardcoded q8 — the caller passes the user's persisted preference as options.defaultQuality
// (precedence rung 3: below the per-(screen,model) sticky, above clamp-to-installed). Absent/invalid
// falls back to q8 (the historical base + worker default), so every existing call site is unchanged.
describe("defaultTierSelection — global defaultQuality setting (sc-10728)", () => {
  it("exposes q8 as the app-wide default and bf16|q8|q4 as the setting's vocabulary", () => {
    expect(DEFAULT_GENERATION_QUALITY).toBe("q8");
    expect(GENERATION_QUALITY_TIERS).toEqual(["bf16", "q8", "q4"]);
  });

  it("defaults the base to q8 when no defaultQuality is supplied", () => {
    const model = matrixModel({ installed: ["q4", "q8", "bf16"], defaultTier: "none" });
    // No options / empty options → the q8 base default (unchanged legacy behavior).
    expect(defaultTierSelection(model, null)).toBe("q8");
    expect(defaultTierSelection(model, null, {})).toBe("q8");
  });

  it("uses the supplied global setting as the base default for a no-sticky model", () => {
    const model = matrixModel({ installed: ["q4", "q8", "bf16"], defaultTier: "none" });
    expect(defaultTierSelection(model, null, { defaultQuality: "bf16" })).toBe("bf16");
    expect(defaultTierSelection(model, null, { defaultQuality: "q4" })).toBe("q4");
    expect(defaultTierSelection(model, null, { defaultQuality: "q8" })).toBe("q8");
  });

  it("applies the global setting to convert-at-install (mlxTiers) models too", () => {
    const model = convertModel({ mlxTiers: ["q4", "q8", "bf16"] });
    expect(defaultTierSelection(model, null, { defaultQuality: "bf16" })).toBe("bf16");
    expect(defaultTierSelection(model, null, { defaultQuality: "q4" })).toBe("q4");
  });

  it("lets a per-(screen,model) sticky (lastUsed) still beat the global setting", () => {
    const model = matrixModel({ installed: ["q4", "q8", "bf16"], defaultTier: "none" });
    // Global setting says bf16, but the user has a sticky q4 for this model → sticky wins (rung 2).
    expect(defaultTierSelection(model, "q4", { defaultQuality: "bf16" })).toBe("q4");
  });

  it("clamps the global setting to installed (bf16 set, only q4 installed → q4)", () => {
    const model = matrixModel({ tiers: ["q4", "q8", "bf16"], installed: ["q4"], defaultTier: "none" });
    expect(defaultTierSelection(model, null, { defaultQuality: "bf16" })).toBe("q4");
  });

  it("falls up from an uninstalled global setting to the nearest clean installed tier", () => {
    // Global setting q4, but only q8 + bf16 are installed → clamp up to the clean q8, never null.
    const model = matrixModel({ tiers: ["q4", "q8", "bf16"], installed: ["q8", "bf16"], defaultTier: "none" });
    expect(defaultTierSelection(model, null, { defaultQuality: "q4" })).toBe("q8");
  });

  it("ignores an invalid global setting and uses the q8 base default", () => {
    const model = matrixModel({ installed: ["q4", "q8", "bf16"], defaultTier: "none" });
    expect(defaultTierSelection(model, null, { defaultQuality: "int8-convrot" })).toBe("q8");
    expect(defaultTierSelection(model, null, { defaultQuality: "bogus" })).toBe("q8");
  });

  it("still honors a manifest-declared default over the global setting", () => {
    // The declared per-model default (rung above the base) is honored even when the global setting differs.
    const model = matrixModel({ installed: ["q4", "q8", "bf16"], defaultTier: "q4" });
    expect(defaultTierSelection(model, null, { defaultQuality: "bf16" })).toBe("q4");
  });

  it("does not let the global setting override an ineligible convrot host filter", () => {
    // defaultQuality can only ever be bf16|q8|q4, so it never re-introduces a hidden convrot tier.
    const model = matrixModel({
      tiers: [INT8_CONVROT_TIER, "bf16"],
      installed: [INT8_CONVROT_TIER, "bf16"],
      defaultTier: "none",
    });
    expect(
      defaultTierSelection(model, null, { defaultQuality: "q4", convRotEligible: false }),
    ).toBe("bf16");
  });
});

// Convert-at-install models (sc-10730): tiers are convert OUTPUTS surfaced as `mlxTiers` (a plain array
// of installed tier keys), NOT the download variant-matrix. The Studio picker reads them so Anima et al.
// get a generation-time tier selector without touching the Models download panel (`hasVariantMatrix`).
function convertModel({ mlxTiers = ["bf16", "q8", "q4"] } = {}) {
  return { id: "anima_base", hasVariantMatrix: false, mlxTiers };
}

describe("quantTier — convert-at-install mlxTiers (sc-10730)", () => {
  it("installedTiers reads mlxTiers, smallest→largest", () => {
    expect(installedTiers(convertModel({ mlxTiers: ["q4", "bf16", "q8"] }))).toEqual([
      "q4",
      "q8",
      "bf16",
    ]);
  });

  it("shows the picker when >1 convert-output tier is present", () => {
    expect(shouldShowTierPicker(convertModel({ mlxTiers: ["bf16", "q8", "q4"] }))).toBe(true);
    expect(shouldShowTierPicker(convertModel({ mlxTiers: ["q8"] }))).toBe(false);
  });

  it("preselects q8 (not q4) so the picker never silently re-sends the washed q4", () => {
    expect(defaultTierSelection(convertModel(), null)).toBe("q8");
    // bf16 when q8 absent (clean-tier fallback), never q4 by default
    expect(defaultTierSelection(convertModel({ mlxTiers: ["bf16", "q4"] }), null)).toBe("bf16");
  });

  it("a last-used convert tier still wins over the q8 default", () => {
    expect(defaultTierSelection(convertModel(), "q4")).toBe("q4");
  });

  it("mlxTiers logic does not disturb download-matrix models (which now also default q8, sc-10726)", () => {
    // A bare non-matrix model with no mlxTiers still surfaces no tiers.
    expect(installedTiers({ id: "x", hasVariantMatrix: false })).toEqual([]);
    // Download-matrix models honor the same app-wide q8 base default (epic 10721 / sc-10726),
    // consistent with the worker resolvers — not the old q4-hard-default.
    expect(
      defaultTierSelection(
        matrixModel({ installed: ["q4", "q8", "bf16"], defaultTier: "none" }),
        null,
      ),
    ).toBe("q8");
  });
});

// Per-model quality floor (sc-10731, epic 10721): the backend surfaces the manifest `mlx.minQualityTier`
// as a top-level `minQualityTier`. `defaultTierSelection` clamps the DEFAULT (rungs 2–4) UP to it — a
// floored model (Anima base/aesthetic = q8) never lets a low global setting / fallback land the default
// on the washed q4 — while an EXPLICIT below-floor picker pick is honored + flagged (`isBelowFloor`).
function flooredConvertModel({ mlxTiers = ["q4", "q8", "bf16"], floor = "q8" } = {}) {
  return { id: "anima_base", hasVariantMatrix: false, mlxTiers, minQualityTier: floor };
}

describe("modelQualityFloor / isBelowFloor (sc-10731)", () => {
  it("reads a valid declared floor and ignores absent/invalid ones", () => {
    expect(modelQualityFloor({ minQualityTier: "q8" })).toBe("q8");
    expect(modelQualityFloor({ minQualityTier: "bf16" })).toBe("bf16");
    expect(modelQualityFloor({ minQualityTier: "q2" })).toBe(null);
    expect(modelQualityFloor({})).toBe(null);
    expect(modelQualityFloor(undefined)).toBe(null);
  });

  it("flags a tier below the floor, not one at/above it", () => {
    const model = { minQualityTier: "q8" };
    expect(isBelowFloor("q4", model)).toBe(true);
    expect(isBelowFloor("q8", model)).toBe(false);
    expect(isBelowFloor("bf16", model)).toBe(false);
    // No floor → nothing is ever below it.
    expect(isBelowFloor("q4", { minQualityTier: undefined })).toBe(false);
    // A non-quality tier (int8-convrot) never participates in a floor compare.
    expect(isBelowFloor(INT8_CONVROT_TIER, model)).toBe(false);
  });
});

describe("defaultTierSelection — per-model quality floor (sc-10731)", () => {
  it("clamps a low global setting UP to the floor (acceptance #1: global q4 + Anima base → q8)", () => {
    const model = flooredConvertModel({ mlxTiers: ["q4", "q8", "bf16"], floor: "q8" });
    // Global "default quality" is q4, but the model floors at q8 → the DEFAULT resolves q8, never q4.
    expect(defaultTierSelection(model, null, { defaultQuality: "q4" })).toBe("q8");
    // The plain q8 base default is already at the floor → still q8.
    expect(defaultTierSelection(model, null)).toBe("q8");
  });

  it("raises a declared default below the floor up to the floor", () => {
    // A download-matrix model that (hypothetically) declares a q4 default but floors at q8 → q8.
    const model = {
      ...matrixModel({ installed: ["q4", "q8", "bf16"], defaultTier: "q4" }),
      minQualityTier: "q8",
    };
    expect(defaultTierSelection(model, null)).toBe("q8");
  });

  it("caps the floor at what's installed (floor q8, only q4 on disk → q4)", () => {
    const model = flooredConvertModel({ mlxTiers: ["q4"], floor: "q8" });
    expect(defaultTierSelection(model, null, { defaultQuality: "q4" })).toBe("q4");
  });

  it("prefers the clean bf16 over the washed q4 when the floor tier is absent", () => {
    // Floor q8 not installed; bf16 (above the floor) + q4 present → the clean-tier fallback picks bf16.
    const model = flooredConvertModel({ mlxTiers: ["q4", "bf16"], floor: "q8" });
    expect(defaultTierSelection(model, null, { defaultQuality: "q4" })).toBe("bf16");
  });

  it("still honors a below-floor STICKY (rung 1) — a prior explicit pick is not re-floored", () => {
    const model = flooredConvertModel({ mlxTiers: ["q4", "q8", "bf16"], floor: "q8" });
    // The user explicitly stickied q4 for this model before → honored as-is (the picker re-flags it).
    expect(defaultTierSelection(model, "q4")).toBe("q4");
  });

  it("leaves non-floored models unaffected (acceptance #3)", () => {
    // No floor (no `minQualityTier`) → the app-wide q8 base default and a q4 global setting both
    // resolve exactly as before.
    const model = { id: "anima_base", hasVariantMatrix: false, mlxTiers: ["q4", "q8", "bf16"] };
    expect(defaultTierSelection(model, null)).toBe("q8");
    expect(defaultTierSelection(model, null, { defaultQuality: "q4" })).toBe("q4");
    // And a bf16 global setting is not lowered by any floor.
    expect(defaultTierSelection(model, null, { defaultQuality: "bf16" })).toBe("bf16");
  });
});

// sc-10732 — acceptance #1 (WEB ANALOG of the worker `default_tier_is_q8_not_q4_regression` guard).
// Epic 10721 moved the app-wide gen-time default OFF the old blind q4 to q8 (sc-10726), matching the
// worker's `preferred_tier(None, None) == "q8"`. If a future change reverts `DEFAULT_GENERATION_QUALITY`
// (or the no-options fallback base) back to q4, these fail loudly on the CI-runnable web side too.
describe("defaultTierSelection — q8-not-q4 base-default revert guard (sc-10732)", () => {
  it("keeps the app-wide base default constant at q8, never q4", () => {
    expect(DEFAULT_GENERATION_QUALITY).toBe("q8");
    expect(DEFAULT_GENERATION_QUALITY).not.toBe("q4");
  });

  it("resolves the no-sticky/no-setting default to q8 (not q4) for matrix AND convert models", () => {
    // Download-matrix model: no sticky, no declared default, no global setting → the q8 base default.
    const matrix = matrixModel({ installed: ["q4", "q8", "bf16"], defaultTier: "none" });
    expect(defaultTierSelection(matrix, null)).toBe("q8");
    expect(defaultTierSelection(matrix, null)).not.toBe("q4");
    // Convert-at-install (mlxTiers) model — the same base default, so the picker never silently
    // re-sends the washed q4 (the sc-10714 Anima quality bug epic 10721 fixed).
    const convert = convertModel({ mlxTiers: ["q4", "q8", "bf16"] });
    expect(defaultTierSelection(convert, null)).toBe("q8");
    expect(defaultTierSelection(convert, null)).not.toBe("q4");
  });
});

// sc-10732 — the FULL precedence LADDER in one place, so the rung ORDER (not just each rung in
// isolation, as the sc-10727/28/31 suites above cover) is locked coherently: peel rungs from the top on
// one floored convert model —
//   sticky (rung 1, un-floored) > global setting (rung 3) > floor clamp > installed clamp > first installed.
describe("defaultTierSelection — full precedence ladder (sc-10732)", () => {
  const floored = () => flooredConvertModel({ mlxTiers: ["q4", "q8", "bf16"], floor: "q8" });

  it("rung 1: a below-floor sticky beats BOTH the global setting and the floor", () => {
    // Sticky q4 (a prior EXPLICIT pick) wins over a bf16 global setting and is NOT re-floored up to q8.
    // Combines two rungs the isolated suites test separately — sticky-beats-global and sticky-below-floor.
    expect(defaultTierSelection(floored(), "q4", { defaultQuality: "bf16" })).toBe("q4");
  });

  it("rung 3: with no sticky, the global setting (above the floor) drives the default", () => {
    expect(defaultTierSelection(floored(), null, { defaultQuality: "bf16" })).toBe("bf16");
  });

  it("floor clamp: a global setting BELOW the floor is raised to the floor, never landing on q4", () => {
    expect(defaultTierSelection(floored(), null, { defaultQuality: "q4" })).toBe("q8");
  });

  it("installed clamp: the floor is capped by what's on disk (floor q8, only q4 installed → q4)", () => {
    const model = flooredConvertModel({ mlxTiers: ["q4"], floor: "q8" });
    expect(defaultTierSelection(model, null, { defaultQuality: "q4" })).toBe("q4");
  });

  it("first-installed fallback: a lone tier outside the clean-fallback chain is the last resort", () => {
    // bf16 alone: no sticky/global/declared/floor; base q8 + fallback [q8,q4] are all absent → the sole
    // installed tier (rung 4 / `tiers[0]`), proving the bottom of the ladder.
    const model = matrixModel({ tiers: ["bf16"], installed: ["bf16"], defaultTier: "none" });
    expect(defaultTierSelection(model, null)).toBe("bf16");
  });
});
