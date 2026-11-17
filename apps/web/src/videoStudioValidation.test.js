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
  implementedMode: true,
  hasInputs: true,
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

  it("surfaces an unimplemented entry point", () => {
    const summary = summarize(videoGenerateValidation({ ...whole, implementedMode: false }));
    expect(summary.surfaced[0].message).toContain("reserved for the next runtime slice");
  });

  it("surfaces the person-replacement worker gate", () => {
    const summary = summarize(videoGenerateValidation({ ...whole, replaceReady: false }));
    expect(summary.surfaced[0].message).toContain("No live GPU worker");
  });

  it("surfaces an execution-tier refusal passed by the active backend lane", () => {
    const message = "Install or repair the accepted tier in Model Manager.";
    const summary = summarize(
      videoGenerateValidation({ ...whole, executionTierBlockMessage: message }),
    );
    expect(summary.ready).toBe(false);
    expect(summary.surfaced.map((entry) => entry.message)).toContain(message);
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
  it("shows the unimplemented entry point without the unfilled-prompt hint beside it", () => {
    const summary = summarize(videoGenerateValidation({ ...whole, prompt: "", implementedMode: false }));
    expect(summary.surfaced).toHaveLength(1);
    expect(summary.surfaced[0].message).toContain("reserved for the next runtime slice");
    expect(summary.ready).toBe(false);
  });
});
