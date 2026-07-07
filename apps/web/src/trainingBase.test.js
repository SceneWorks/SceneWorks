import { describe, expect, it } from "vitest";
import { trainingBaseState, trainingBaseTier } from "./trainingBase.js";

// A /models-shaped base with per-variant install states (sc-8508). `installState` is the string the
// catalog emits per tier ("installed" / "missing").
function base(variants, installState = "installed") {
  return { installState, variants };
}
const v = (variant, installState) => ({ variant, installState });

describe("trainingBaseTier", () => {
  it("prefers a dedicated `training` variant (lens, sc-8797) over the bf16 inference tier", () => {
    // Lens ships q4/q8/bf16 MLX inference tiers AND a separate flat-diffusers training base.
    const lens = base([v("q4", "installed"), v("q8", "missing"), v("bf16", "missing"), v("training", "missing")]);
    expect(trainingBaseTier(lens)).toBe("training");
  });

  it("falls back to the dense bf16 tier when there is no training variant (Krea 2 Raw, epic 9992)", () => {
    const krea = base([v("q4", "installed"), v("q8", "installed"), v("bf16", "missing")]);
    expect(trainingBaseTier(krea)).toBe("bf16");
  });

  it("returns undefined for a non-matrix base (z-image / sdxl — trains on its default tier)", () => {
    expect(trainingBaseTier(base(undefined))).toBe(undefined);
    expect(trainingBaseTier(base([]))).toBe(undefined);
    expect(trainingBaseTier(base([v("q4", "installed"), v("q8", "missing")]))).toBe(undefined);
    expect(trainingBaseTier(undefined)).toBe(undefined);
  });
});

describe("trainingBaseState", () => {
  it("reports the TRAINING variant's own state — lens q4-installed but training-missing reads missing (the sc-8966 bug)", () => {
    const lens = base(
      [v("q4", "installed"), v("q8", "missing"), v("bf16", "missing"), v("training", "missing")],
      // Top-level installState is "installed" (reflects the default q4 tier) — the OLD gate read this and
      // wrongly said "ready". The fix reads the training variant instead.
      "installed",
    );
    expect(trainingBaseState(lens)).toBe("missing");
  });

  it("reports installed once the training base itself is present", () => {
    const lens = base([v("q4", "installed"), v("training", "installed")], "installed");
    expect(trainingBaseState(lens)).toBe("installed");
  });

  it("reports the bf16 tier's state for a training-variant-less matrix base (Krea)", () => {
    expect(trainingBaseState(base([v("q4", "installed"), v("bf16", "missing")]))).toBe("missing");
    expect(trainingBaseState(base([v("q4", "installed"), v("bf16", "installed")]))).toBe("installed");
  });

  it("falls back to the top-level installState for a non-matrix base", () => {
    expect(trainingBaseState(base([v("q4", "installed")], "installed"))).toBe("installed");
    expect(trainingBaseState(base(undefined, "missing"))).toBe("missing");
  });

  it("treats a training tier declared but absent from variants as missing (never a false ready)", () => {
    // Defensive: trainingBaseTier says "training" but no matching entry carries a state.
    const weird = { installState: "installed", variants: [{ variant: "training" }] };
    expect(trainingBaseState(weird)).toBe("missing");
  });
});
