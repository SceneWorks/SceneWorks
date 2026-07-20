import { describe, expect, it } from "vitest";

import { summarize } from "./validation/issues.js";
import { videoGenerateValidation } from "./videoStudioValidation.js";
import { PROMPT_MAX_CHARS } from "./styleComposer.js";

// sc-13136 — the composed-prompt budget guard mirrored from the Image Studio (sc-13133). A catalog
// style wraps ~700–900 chars AROUND the user's prompt, so a long-but-under-cap prompt can compose
// past the backend cap. The gate measures the COMPOSED string and ONLY fires when a style is active;
// styleless behavior is unchanged.

const whole = {
  activeProject: { id: "p1" },
  promptless: false,
  prompt: "a dog running",
  supportsMode: true,
  implementedMode: true,
  hasInputs: true,
  requiresLtxIcLora: false,
  hasLtxIcLora: false,
  replaceReady: true,
  modelName: "Wan",
  presetMissing: [],
  presetIncompatible: [],
  loraIncompatible: [],
};

const errorMessages = (issues) => issues.filter((i) => i.kind === "error").map((i) => i.message);

describe("videoGenerateValidation — Style Catalog budget gate (sc-13136)", () => {
  it("no style active → the composed length is never measured (styleless unchanged)", () => {
    const overLong = "x".repeat(PROMPT_MAX_CHARS + 500);
    const issues = videoGenerateValidation({ ...whole, styleActive: false, composedPrompt: overLong });
    expect(summarize(issues).ready).toBe(true);
    expect(errorMessages(issues)).toEqual([]);
  });

  it("style active + composed prompt within budget → ready", () => {
    const composed = `Subject: ${"x".repeat(200)}\nStyle: a gentle style`;
    const issues = videoGenerateValidation({ ...whole, styleActive: true, composedPrompt: composed });
    expect(summarize(issues).ready).toBe(true);
    expect(errorMessages(issues)).toEqual([]);
  });

  it("style active + composed prompt OVER budget → a blocking error naming the count", () => {
    const composed = "x".repeat(PROMPT_MAX_CHARS + 1);
    const issues = videoGenerateValidation({ ...whole, styleActive: true, composedPrompt: composed });
    expect(summarize(issues).ready).toBe(false);
    const errors = errorMessages(issues);
    expect(errors.length).toBe(1);
    expect(errors[0]).toContain(`${PROMPT_MAX_CHARS + 1}/${PROMPT_MAX_CHARS}`);
    expect(errors[0]).toMatch(/shorten your prompt or pick a shorter style/);
  });

  it("boundary: exactly at the cap is allowed; one over is not", () => {
    const atCap = "x".repeat(PROMPT_MAX_CHARS);
    const overCap = "x".repeat(PROMPT_MAX_CHARS + 1);
    expect(summarize(videoGenerateValidation({ ...whole, styleActive: true, composedPrompt: atCap })).ready).toBe(true);
    expect(summarize(videoGenerateValidation({ ...whole, styleActive: true, composedPrompt: overCap })).ready).toBe(false);
  });
});
