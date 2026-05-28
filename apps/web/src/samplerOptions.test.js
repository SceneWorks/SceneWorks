import { describe, expect, it } from "vitest";
import {
  SAMPLER_LABELS,
  SCHEDULER_LABELS,
  guidanceDefaultFromModel,
  samplerDefaultFromModel,
  samplerOptionsFromModel,
  schedulerDefaultFromModel,
  schedulerOptionsFromModel,
  schedulerShiftDefaultFromModel,
  stepsDefaultFromModel,
} from "./samplerOptions.js";

describe("samplerOptions", () => {
  it("falls back to default-only when limits are missing", () => {
    expect(samplerOptionsFromModel(undefined)).toEqual(["default"]);
    expect(samplerOptionsFromModel({})).toEqual(["default"]);
    expect(schedulerOptionsFromModel({})).toEqual(["default"]);
  });

  it("returns options in canonical order regardless of manifest ordering", () => {
    const model = { limits: { samplers: ["unipc", "default", "euler"] } };
    expect(samplerOptionsFromModel(model)).toEqual(["default", "euler", "unipc"]);
  });

  it("preserves unknown sampler keys (forward-compat)", () => {
    const model = { limits: { samplers: ["default", "euler", "future"] } };
    expect(samplerOptionsFromModel(model)).toEqual(["default", "euler", "future"]);
  });

  it("scheduler options are ordered canonically", () => {
    const model = {
      limits: {
        schedulers: ["beta", "default", "karras", "shift"],
      },
    };
    expect(schedulerOptionsFromModel(model)).toEqual(["default", "shift", "karras", "beta"]);
  });

  it("default helpers read defaults block with sensible fallbacks", () => {
    const model = {
      defaults: {
        sampler: "dpmpp",
        scheduler: "karras",
        schedulerShift: 4.5,
        steps: 20,
        guidanceScale: 3.5,
      },
    };
    expect(samplerDefaultFromModel(model)).toBe("dpmpp");
    expect(schedulerDefaultFromModel(model)).toBe("karras");
    expect(schedulerShiftDefaultFromModel(model)).toBe(4.5);
    expect(stepsDefaultFromModel(model)).toBe(20);
    expect(guidanceDefaultFromModel(model)).toBe(3.5);
  });

  it("invalid default values fall back gracefully", () => {
    expect(samplerDefaultFromModel({ defaults: { sampler: "" } })).toBe("default");
    expect(schedulerShiftDefaultFromModel({ defaults: { schedulerShift: -1 } })).toBe(3.0);
    expect(stepsDefaultFromModel({ defaults: { steps: 0 } })).toBeNull();
    expect(guidanceDefaultFromModel({ defaults: { guidanceScale: "n/a" } })).toBeNull();
  });

  it("labels cover the full menu", () => {
    for (const key of ["default", "euler", "heun", "dpmpp", "unipc"]) {
      expect(SAMPLER_LABELS[key]).toBeTruthy();
    }
    for (const key of ["default", "simple", "shift", "karras", "exponential", "beta"]) {
      expect(SCHEDULER_LABELS[key]).toBeTruthy();
    }
  });
});
