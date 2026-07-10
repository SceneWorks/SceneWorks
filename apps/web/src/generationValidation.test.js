import { describe, expect, it } from "vitest";

import { presetLoraIssues } from "./generationValidation.js";
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
