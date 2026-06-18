import { describe, expect, it } from "vitest";

import { modelUsesGuidance, resolveCreativityGuidance } from "./simpleDefaults.js";

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
