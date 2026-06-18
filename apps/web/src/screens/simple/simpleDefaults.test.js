import { describe, expect, it } from "vitest";

import { FALLBACK_DURATIONS, modelUsesGuidance, resolveCreativityGuidance, videoDurations } from "./simpleDefaults.js";

const turbo = { id: "z_image_turbo", defaults: { steps: 8, guidanceScale: 0 } };
const base = { id: "realvisxl", defaults: { steps: 30, guidanceScale: 5 } };

describe("Creativity → guidance (model-aware)", () => {
  it("is hidden for distilled/turbo models that ignore guidance", () => {
    expect(modelUsesGuidance(turbo)).toBe(false);
    expect(resolveCreativityGuidance(turbo, "close")).toBe(null);
    expect(resolveCreativityGuidance(turbo, "creative")).toBe(null);
  });

  it("applies for base models that honor guidance", () => {
    expect(modelUsesGuidance(base)).toBe(true);
    expect(resolveCreativityGuidance(base, "balanced")).toBe(null); // default
    expect(resolveCreativityGuidance(base, "close")).toBe(7); // round(5 * 1.4)
    expect(resolveCreativityGuidance(base, "creative")).toBe(3); // round(5 * 0.6)
  });

  it("clamps to a sane 1-12 band", () => {
    expect(resolveCreativityGuidance({ defaults: { guidanceScale: 1 } }, "creative")).toBe(1); // 0.6 → 1
    expect(resolveCreativityGuidance({ defaults: { guidanceScale: 10 } }, "close")).toBe(12); // 14 → 12
  });
});

describe("videoDurations (per-model, capped at recommended)", () => {
  const ltx = { limits: { durations: [4, 6, 8, 10, 12, 15], recommendedMaxDuration: 10, hardMaxDuration: 15 } };

  it("caps at the model's recommendedMaxDuration", () => {
    expect(videoDurations(ltx)).toEqual([4, 6, 8, 10]);
  });

  it("returns the full list when no recommended cap is declared", () => {
    expect(videoDurations({ limits: { durations: [4, 6, 8, 10, 12, 15] } })).toEqual([4, 6, 8, 10, 12, 15]);
  });

  it("ignores a cap below the smallest option (never empties the list)", () => {
    expect(videoDurations({ limits: { durations: [8, 10], recommendedMaxDuration: 4 } })).toEqual([8, 10]);
  });

  it("falls back to the shared list when the model declares no durations", () => {
    expect(videoDurations(null)).toEqual(FALLBACK_DURATIONS);
    expect(videoDurations({ limits: {} })).toEqual(FALLBACK_DURATIONS);
  });
});
