import { describe, expect, it } from "vitest";

import { modelUsesGuidance, resolveCreativityGuidance, resolveDetailSteps } from "./simpleDefaults.js";

const turbo = { id: "z_image_turbo", defaults: { steps: 8, guidanceScale: 0 } };
const base = { id: "realvisxl", defaults: { steps: 30, guidanceScale: 5 } };
const noDefaults = { id: "mystery", defaults: {} };

describe("Detail → steps", () => {
  it("omits (null) at Standard so the model default is used", () => {
    expect(resolveDetailSteps(turbo, "standard")).toBe(null);
    expect(resolveDetailSteps(base, "standard")).toBe(null);
  });

  it("scales over the model's own default", () => {
    expect(resolveDetailSteps(turbo, "high")).toBe(11); // round(8 * 1.4)
    expect(resolveDetailSteps(turbo, "max")).toBe(15); // round(8 * 1.85)
    expect(resolveDetailSteps(base, "high")).toBe(42); // round(30 * 1.4)
  });

  it("falls back to a safe anchor when the model has no steps default", () => {
    expect(resolveDetailSteps(noDefaults, "high")).toBe(28); // round(20 * 1.4)
  });

  it("clamps to the worker's 1-80 ceiling", () => {
    expect(resolveDetailSteps({ defaults: { steps: 60 } }, "max")).toBe(80); // 60*1.85=111 → 80
  });
});

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
