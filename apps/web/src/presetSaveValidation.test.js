import { describe, expect, it } from "vitest";

import { presetSaveValidation } from "./presetUtils.js";
import { summarize } from "./validation/issues.js";

// The Preset Manager Save gate (epic 10651). Replaces the saveDisabledReason ternary and
// presetValidationMessage. Name/model are silent requirements; a read-only built-in and
// the preset's own LoRA problems are surfaced errors.
describe("presetSaveValidation", () => {
  const whole = { editable: true, name: "My Preset", model: "z_image_turbo" };
  const clean = { validation: { missing: [], incompatible: [], ok: true } };
  const kinds = (issues, field) => issues.filter((i) => i.field === field).map((i) => i.kind);

  it("passes a whole editable preset", () => {
    expect(summarize(presetSaveValidation(whole, clean)).ready).toBe(true);
    expect(summarize(presetSaveValidation(whole, clean)).surfaced).toEqual([]);
  });

  it("surfaces a read-only built-in, and only that", () => {
    const issues = presetSaveValidation({ editable: false, name: "", model: "" }, clean);
    expect(issues).toHaveLength(1);
    expect(issues[0].kind).toBe("error");
    expect(issues[0].message).toContain("Built-in presets are read-only");
  });

  it("surfaces each broken default value (valueErrors) as its own error", () => {
    const summary = summarize(
      presetSaveValidation(whole, { validation: { missing: [], incompatible: [] }, valueErrors: ["Steps must be a whole number between 1 and 200.", "Aspect 99x99 isn't one this model supports — pick a listed option."] }),
    );
    expect(summary.surfaced).toHaveLength(2);
    expect(summary.surfaced.every((i) => i.kind === "error")).toBe(true);
    expect(summary.surfaced[0].message).toContain("Steps must be");
    expect(summary.surfaced[1].message).toContain("isn't one this model supports");
    expect(summary.ready).toBe(false);
  });

  it("keeps name and model silent", () => {
    const issues = presetSaveValidation({ editable: true, name: "  ", model: "" }, clean);
    expect(kinds(issues, "name")).toEqual(["requirement"]);
    expect(kinds(issues, "model")).toEqual(["requirement"]);
    expect(summarize(issues).surfaced).toEqual([]);
    expect(summarize(issues).ready).toBe(false);
  });

  it("surfaces a preset LoRA still importing, preserving the recognizable copy", () => {
    const summary = summarize(presetSaveValidation(whole, { validation: { missing: ["pending_style"], incompatible: [] } }));
    expect(summary.surfaced).toHaveLength(1);
    expect(summary.surfaced[0].kind).toBe("error");
    // App.videoPresets asserts this exact substring.
    expect(summary.surfaced[0].message).toContain("Save blocked: pending_style has not finished importing.");
  });

  it("surfaces incompatible preset LoRAs", () => {
    const summary = summarize(presetSaveValidation(whole, { validation: { missing: [], incompatible: ["sd15_lora"] } }));
    expect(summary.surfaced[0].message).toContain("sd15_lora");
    expect(summary.surfaced[0].message).toContain("not compatible with the selected model");
  });

  it("pluralizes both LoRA messages", () => {
    const summary = summarize(presetSaveValidation(whole, { validation: { missing: ["a", "b"], incompatible: ["c", "d"] } }));
    expect(summary.surfaced[0].message).toContain("a, b have not finished importing");
    expect(summary.surfaced[1].message).toContain("c, d are not compatible");
  });

  it("tolerates a missing validation context", () => {
    expect(() => presetSaveValidation(whole)).not.toThrow();
    expect(summarize(presetSaveValidation(whole)).ready).toBe(true);
  });
});
