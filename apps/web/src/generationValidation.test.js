import { describe, expect, it } from "vitest";

import { presetLoraIssues, savePresetDialogValidation } from "./generationValidation.js";
import { summarize } from "./validation/issues.js";

// The preset/LoRA problems shared by both studios' Generate gates (epic 10650). Pinned
// once here so the three messages have a single home; the studio rule tests then trust it.
describe("presetLoraIssues", () => {
  it("is empty when nothing is wrong", () => {
    expect(presetLoraIssues({})).toEqual([]);
    expect(presetLoraIssues()).toEqual([]);
  });

  it("emits an error per problem, all form-scoped", () => {
    const issues = presetLoraIssues({
      presetMissing: ["a"],
      presetIncompatible: ["b"],
      loraIncompatible: ["c"],
      modelName: "FLUX",
    });
    expect(issues.map((i) => i.kind)).toEqual(["error", "error", "error"]);
    expect(issues.every((i) => i.field === null)).toBe(true);
    expect(summarize(issues).surfaced).toHaveLength(3);
  });

  it("keeps the three messages distinct and names the model where relevant", () => {
    const [missing] = presetLoraIssues({ presetMissing: ["a"] });
    expect(missing.message).toContain("until LoRA import finishes");
    const [incompat] = presetLoraIssues({ presetIncompatible: ["b"], modelName: "Qwen" });
    expect(incompat.message).toContain("Qwen");
    const [lora] = presetLoraIssues({ loraIncompatible: ["c"], modelName: undefined });
    expect(lora.message).toContain("the selected model");
  });
});

// The studios' inline "Save as Preset" dialog (epic 10651): a blank name is silent, an
// unsaveable mode surfaces the caller's tooltip as an always-visible chip.
describe("savePresetDialogValidation", () => {
  it("passes a named, saveable setup", () => {
    expect(summarize(savePresetDialogValidation({ presetName: "My look", saveDisabled: false })).ready).toBe(true);
  });

  it("requires a name, silently", () => {
    const summary = summarize(savePresetDialogValidation({ presetName: "   ", saveDisabled: false }));
    expect(summary.ready).toBe(false);
    expect(summary.surfaced).toEqual([]);
  });

  it("surfaces the tooltip when the mode can't be saved", () => {
    const summary = summarize(
      savePresetDialogValidation({ presetName: "My look", saveDisabled: true, saveTitle: "Presets are available in Image→Video mode." }),
    );
    expect(summary.surfaced).toHaveLength(1);
    expect(summary.surfaced[0].kind).toBe("error");
    expect(summary.surfaced[0].message).toBe("Presets are available in Image→Video mode.");
  });

  it("falls back to a default message when no tooltip is provided", () => {
    const summary = summarize(savePresetDialogValidation({ presetName: "x", saveDisabled: true }));
    expect(summary.surfaced[0].message).toContain("can’t be saved as a preset");
  });
});
