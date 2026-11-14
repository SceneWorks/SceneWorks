import { describe, expect, it } from "vitest";
import { generationRefusal, jobFailureMessage } from "./generationRefusal.js";

describe("generation refusal presentation", () => {
  it("keeps field and context for HTTP and queued refusals without touching memory telemetry", () => {
    const refusal = { code: "invalid_generation_option", detail: "Choose a supported rate", field: "fps", context: { model: "mochi_1" } };
    const job = { error: refusal.detail, result: { validationRefusal: refusal, memoryStrategy: { rung: "resident" } } };
    expect(generationRefusal(job)).toEqual(refusal);
    expect(generationRefusal(refusal)).toEqual(refusal);
    expect(jobFailureMessage(job)).toBe("fps: Choose a supported rate");
    expect(job.result.memoryStrategy).toEqual({ rung: "resident" });
  });
  it("renders training and Yue2 typed refusals", () => {
    for (const code of ["training_target_limit", "InvalidField"]) {
      const refusal = { code, detail: "Choose a smaller value", context: { field: "rank", maximum: 128 } };
      expect(jobFailureMessage({ result: { validationRefusal: refusal } })).toBe("rank: Choose a smaller value");
    }
  });
  it("preserves historical plain failures and does not repeat named fields", () => {
    expect(jobFailureMessage({ error: "GPU unavailable" })).toBe("GPU unavailable");
    expect(jobFailureMessage({ result: { validationRefusal: { code: "invalid_generation_option", field: "width", detail: "width must be an integer" } } })).toBe("width must be an integer");
  });
});
