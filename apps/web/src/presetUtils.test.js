import { describe, expect, it } from "vitest";
import { MAX_JOB_LORAS_TOTAL, MAX_PRESET_LORAS, MAX_USER_JOB_LORAS } from "./presetUtils.js";

// sc-8936 (F-134): the LoRA caps are now single constants instead of scattered magic
// numbers. These assertions pin the values + the intended relationship so a future
// change to one has to consciously reconcile the others (and keep them in sync with the
// worker guards documented alongside the constants).
describe("LoRA cap constants (sc-8936)", () => {
  it("matches the current cap values", () => {
    expect(MAX_JOB_LORAS_TOTAL).toBe(5);
    expect(MAX_USER_JOB_LORAS).toBe(4);
    expect(MAX_PRESET_LORAS).toBe(5);
  });

  it("leaves headroom for one auto-applied builtin within the per-job total", () => {
    // The studio pickers cap user-selectable LoRAs below the hard total so a builtin
    // can still ride along (pick 4 user -> total 5), matching the worker guard.
    expect(MAX_USER_JOB_LORAS).toBeLessThan(MAX_JOB_LORAS_TOTAL);
    expect(MAX_USER_JOB_LORAS + 1).toBe(MAX_JOB_LORAS_TOTAL);
  });

  it("caps saved-preset LoRAs at the per-job total", () => {
    expect(MAX_PRESET_LORAS).toBe(MAX_JOB_LORAS_TOTAL);
  });
});
