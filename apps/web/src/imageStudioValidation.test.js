import { describe, expect, it } from "vitest";

import { summarize } from "./validation/issues.js";
import { imageBatchValidation, imageGenerateValidation } from "./imageStudioValidation.js";

// The two Image Studio CTA gates in the app-wide vocabulary (epic 10649). The kinds are the
// contract: a requirement blocks in silence (the empty field shows it), an error blocks and
// says why. Both rule sets are driven through summarize() so a mis-kinded condition fails
// here rather than passing on a fixture that happens to agree with it.

describe("imageGenerateValidation", () => {
  const whole = {
    activeProject: { id: "p1" },
    structuredActive: false,
    prompt: "a cat",
    mode: "text_image",
    presetMissing: [],
    presetIncompatible: [],
    loraIncompatible: [],
    modelName: "FLUX",
  };
  const kinds = (issues, field) => issues.filter((i) => i.field === field).map((i) => i.kind);

  it("passes a whole plain-text draft", () => {
    expect(summarize(imageGenerateValidation(whole)).ready).toBe(true);
    expect(summarize(imageGenerateValidation(whole)).surfaced).toEqual([]);
  });

  it("requires a project, silently", () => {
    const issues = imageGenerateValidation({ ...whole, activeProject: null });
    expect(kinds(issues, "project")).toEqual(["requirement"]);
    expect(summarize(issues).surfaced).toEqual([]);
    expect(summarize(issues).ready).toBe(false);
  });

  it("requires a plain prompt, silently", () => {
    const issues = imageGenerateValidation({ ...whole, prompt: "   " });
    expect(kinds(issues, "prompt")).toEqual(["requirement"]);
    expect(summarize(issues).surfaced).toEqual([]);
  });

  it("requires caption content on a structured model, silently", () => {
    const issues = imageGenerateValidation({ ...whole, structuredActive: true, captionHasContent: false, prompt: "" });
    expect(kinds(issues, "caption")).toEqual(["requirement"]);
    // The plain-prompt rule must not also fire on a structured model.
    expect(kinds(issues, "prompt")).toEqual([]);
    expect(summarize(issues).surfaced).toEqual([]);
  });

  it("requires a character in character mode, silently", () => {
    const issues = imageGenerateValidation({ ...whole, mode: "character_image", characterId: "" });
    expect(kinds(issues, "character")).toEqual(["requirement"]);
    expect(summarize(issues).surfaced).toEqual([]);
  });

  it("requires a source image in edit mode, silently (epic 10871)", () => {
    const issues = imageGenerateValidation({ ...whole, mode: "edit_image", editSourceMissing: true });
    expect(kinds(issues, "source")).toEqual(["requirement"]);
    expect(summarize(issues).surfaced).toEqual([]);
    expect(summarize(issues).ready).toBe(false);
  });

  it("does not require a source once one is picked", () => {
    const issues = imageGenerateValidation({ ...whole, mode: "edit_image", editSourceMissing: false });
    expect(kinds(issues, "source")).toEqual([]);
  });

  it("gates on a missing edit LoRA silently — the source band carries the download note (epic 10871)", () => {
    const issues = imageGenerateValidation({ ...whole, mode: "edit_image", editLoraMissing: true });
    expect(kinds(issues, "editLora")).toEqual(["requirement"]);
    expect(summarize(issues).surfaced).toEqual([]);
    expect(summarize(issues).ready).toBe(false);
  });

  it("surfaces preset-missing, preset-incompatible and lora-incompatible as errors", () => {
    const issues = imageGenerateValidation({
      ...whole,
      presetMissing: ["loraA"],
      presetIncompatible: ["loraB"],
      loraIncompatible: ["loraC"],
    });
    const summary = summarize(issues);
    expect(summary.surfaced).toHaveLength(3);
    expect(summary.surfaced.every((i) => i.kind === "error")).toBe(true);
    expect(summary.surfaced[0].message).toContain("loraA");
    expect(summary.surfaced[1].message).toContain("FLUX");
    expect(summary.surfaced[1].message).toContain("loraB");
    expect(summary.surfaced[2].message).toContain("loraC");
    expect(summary.ready).toBe(false);
  });

  it("names the selected model, or a fallback when unknown", () => {
    const named = imageGenerateValidation({ ...whole, presetIncompatible: ["x"], modelName: "Qwen" });
    expect(summarize(named).surfaced[0].message).toContain("Qwen");
    const anon = imageGenerateValidation({ ...whole, presetIncompatible: ["x"], modelName: undefined });
    expect(summarize(anon).surfaced[0].message).toContain("the selected model");
  });

  // Bidirectional (epic contract). Direction 2: a broken value shows alone, the unfilled
  // field beside it stays silent — pinned exactly, not with toContain.
  it("shows the broken preset without the unfilled-project hint", () => {
    const summary = summarize(imageGenerateValidation({ ...whole, activeProject: null, presetMissing: ["loraA"] }));
    expect(summary.surfaced).toHaveLength(1);
    expect(summary.surfaced[0].message).toContain("loraA");
  });
});

describe("imageBatchValidation", () => {
  const whole = {
    activeProject: { id: "p1" },
    batchStructuredExpandBlocked: false,
    batchTotal: 4,
    missingKeys: [],
    groupIssues: [],
    resolutionIssues: [],
    minDimension: 256,
    maxDimension: 4096,
  };

  it("passes a whole batch draft", () => {
    expect(summarize(imageBatchValidation(whole)).ready).toBe(true);
    expect(summarize(imageBatchValidation(whole)).surfaced).toEqual([]);
  });

  it("treats an empty batch as a silent requirement", () => {
    const issues = imageBatchValidation({ ...whole, batchTotal: 0 });
    expect(issues.find((i) => i.field === "prompts").kind).toBe("requirement");
    expect(summarize(issues).surfaced).toEqual([]);
    expect(summarize(issues).ready).toBe(false);
  });

  it("surfaces missing template keys as an error", () => {
    const summary = summarize(imageBatchValidation({ ...whole, missingKeys: ["color", "size"] }));
    expect(summary.surfaced[0].kind).toBe("error");
    expect(summary.surfaced[0].message).toBe("Fill in a value for {{color}}, {{size}} to run.");
  });

  it("surfaces an out-of-range resolution with the offending size", () => {
    const summary = summarize(imageBatchValidation({ ...whole, resolutionIssues: [{ width: 5000, height: 300 }] }));
    expect(summary.surfaced[0].message).toBe("A prompt’s [5000×300] size is out of range — each side must be 256–4096.");
  });

  // The batch panel renders surfaced[0]; the rules must push in the same priority order the
  // old ?: chain used, or the wrong message wins.
  it("orders the errors: expand-blocked, missing keys, group, resolution", () => {
    const summary = summarize(
      imageBatchValidation({
        ...whole,
        batchStructuredExpandBlocked: true,
        missingKeys: ["k"],
        groupIssues: [{ label: "g" }],
        resolutionIssues: [{ width: 9000, height: 9000 }],
      }),
    );
    expect(summary.surfaced[0].message).toContain("prompt-refiner model installed");
    expect(summary.surfaced[1].message).toContain("Fill in a value");
    expect(summary.surfaced[2].message).toContain("same number of options");
    expect(summary.surfaced[3].message).toContain("out of range");
  });

  it("keeps the batch's problems out of a project requirement's way", () => {
    const issues = imageBatchValidation({ ...whole, activeProject: null });
    expect(issues.find((i) => i.field === "project").kind).toBe("requirement");
    expect(summarize(issues).ready).toBe(false);
  });
});
