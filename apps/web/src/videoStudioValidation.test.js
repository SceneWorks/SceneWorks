import { describe, expect, it } from "vitest";

import { summarize } from "./validation/issues.js";
import { videoGenerateValidation } from "./videoStudioValidation.js";

// The Video Studio Generate gate (epic 10650). This screen carried the epic's cleanest
// drift bug — `canSubmit` and `blockedMessage` re-deriving the same rules side by side —
// so the tests lean on the two-directions contract: a broken value surfaces its own
// reason, and an unfilled field stays silent.

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

const kinds = (issues, field) => issues.filter((i) => i.field === field).map((i) => i.kind);

describe("videoGenerateValidation", () => {
  it("passes a whole draft", () => {
    expect(summarize(videoGenerateValidation(whole)).ready).toBe(true);
    expect(summarize(videoGenerateValidation(whole)).surfaced).toEqual([]);
  });

  it("requires project / prompt / inputs, all silent", () => {
    const issues = videoGenerateValidation({ ...whole, activeProject: null, prompt: "", hasInputs: false });
    expect(kinds(issues, "project")).toEqual(["requirement"]);
    expect(kinds(issues, "prompt")).toEqual(["requirement"]);
    expect(kinds(issues, "inputs")).toEqual(["requirement"]);
    expect(summarize(issues).surfaced).toEqual([]);
    expect(summarize(issues).ready).toBe(false);
  });

  it("does not gate on prompt for a promptless model", () => {
    const issues = videoGenerateValidation({ ...whole, promptless: true, prompt: "" });
    expect(kinds(issues, "prompt")).toEqual([]);
    expect(summarize(issues).ready).toBe(true);
  });

  it("surfaces an unsupported mode, naming the model", () => {
    const summary = summarize(videoGenerateValidation({ ...whole, supportsMode: false }));
    expect(summary.surfaced).toHaveLength(1);
    expect(summary.surfaced[0].kind).toBe("error");
    expect(summary.surfaced[0].message).toBe("Wan does not support this mode.");
  });

  it("falls back to 'Selected model' when the model is unnamed", () => {
    const summary = summarize(videoGenerateValidation({ ...whole, supportsMode: false, modelName: undefined }));
    expect(summary.surfaced[0].message).toBe("Selected model does not support this mode.");
  });

  it("surfaces an unimplemented entry point", () => {
    const summary = summarize(videoGenerateValidation({ ...whole, implementedMode: false }));
    expect(summary.surfaced[0].message).toContain("reserved for the next runtime slice");
  });

  it("surfaces the LTX IC-LoRA requirement only when required and absent", () => {
    expect(summarize(videoGenerateValidation({ ...whole, requiresLtxIcLora: true, hasLtxIcLora: false })).surfaced[0].message).toContain(
      "IC-LoRA preset",
    );
    // Required but present → no issue.
    expect(summarize(videoGenerateValidation({ ...whole, requiresLtxIcLora: true, hasLtxIcLora: true })).ready).toBe(true);
    // Not required → no issue even when absent.
    expect(summarize(videoGenerateValidation({ ...whole, requiresLtxIcLora: false, hasLtxIcLora: false })).ready).toBe(true);
  });

  it("surfaces the person-replacement worker gate", () => {
    const summary = summarize(videoGenerateValidation({ ...whole, replaceReady: false }));
    expect(summary.surfaced[0].message).toContain("No live GPU worker");
  });

  it("surfaces preset and selected-LoRA problems", () => {
    const summary = summarize(
      videoGenerateValidation({ ...whole, presetMissing: ["a"], presetIncompatible: ["b"], loraIncompatible: ["c"] }),
    );
    expect(summary.surfaced.map((i) => i.kind)).toEqual(["error", "error", "error"]);
    expect(summary.surfaced[2].message).toContain("Wan");
  });

  // Bidirectional: a broken value shows its reason; the unfilled field beside it stays
  // silent. This is the exact pairing the old blockedMessage/canSubmit split kept drifting.
  it("shows the mode error without the unfilled-prompt hint beside it", () => {
    const summary = summarize(videoGenerateValidation({ ...whole, prompt: "", supportsMode: false }));
    expect(summary.surfaced).toHaveLength(1);
    expect(summary.surfaced[0].message).toContain("does not support this mode");
    expect(summary.ready).toBe(false);
  });
});
