import { describe, expect, it } from "vitest";

import { summarize } from "./validation/issues.js";
import {
  batchPromptBudgetOverages,
  imageBatchValidation,
  imageGenerateValidation,
} from "./imageStudioValidation.js";
import { composeStyledPrompt, PROMPT_MAX_CHARS } from "./styleComposer.js";

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

  // Composed-prompt budget guard (sc-13133). The cap is measured on the COMPOSED outgoing prompt,
  // not the raw prompt field, and only when a style is active — styleless behavior is unchanged.
  describe("composed-prompt budget (sc-13133)", () => {
    // A raw prompt that is itself well UNDER the cap, but a style long enough that the composed
    // Subject:/Style: string runs OVER it. The whole point of the guard.
    const rawPrompt = "a fox in the snow"; // 17 chars — nowhere near the cap on its own
    const longStyle = "x".repeat(PROMPT_MAX_CHARS); // composed will exceed the cap by the wrapper + prompt
    const overComposed = composeStyledPrompt({ styleText: longStyle, userPrompt: rawPrompt });

    it("styleless is unchanged: an over-cap composedPrompt is ignored when no style is active", () => {
      // Even handed an over-cap string, a styleless draft must not sprout a new error — the raw
      // prompt keeps whatever gating it had (the backend still bounds it).
      const issues = imageGenerateValidation({ ...whole, styleActive: false, composedPrompt: overComposed });
      expect(summarize(issues).surfaced).toEqual([]);
      expect(summarize(issues).ready).toBe(true);
    });

    it("under budget with a style active surfaces no error", () => {
      const underComposed = composeStyledPrompt({ styleText: "cinematic watercolor", userPrompt: rawPrompt });
      expect([...underComposed].length).toBeLessThanOrEqual(PROMPT_MAX_CHARS);
      const issues = imageGenerateValidation({ ...whole, styleActive: true, composedPrompt: underComposed });
      expect(summarize(issues).surfaced).toEqual([]);
      expect(summarize(issues).ready).toBe(true);
    });

    // DISCRIMINATION: raw prompt < cap, composed > cap → the guard MUST fire. A guard that measured
    // the raw `prompt` field (17 chars) instead of the composed string would let this through — this
    // test fails that mutation.
    it("blocks and surfaces an error when the COMPOSED prompt exceeds the cap (raw prompt is short)", () => {
      expect([...rawPrompt].length).toBeLessThan(PROMPT_MAX_CHARS); // raw is well under
      expect([...overComposed].length).toBeGreaterThan(PROMPT_MAX_CHARS); // composed is over
      const issues = imageGenerateValidation({
        ...whole,
        prompt: rawPrompt,
        styleActive: true,
        composedPrompt: overComposed,
      });
      const summary = summarize(issues);
      expect(summary.ready).toBe(false);
      expect(summary.surfaced).toHaveLength(1);
      expect(summary.surfaced[0].kind).toBe("error");
      // The message names the real numbers and the two ways out — no raw internals.
      expect(summary.surfaced[0].message).toContain(`${[...overComposed].length}/${PROMPT_MAX_CHARS}`);
      expect(summary.surfaced[0].message).toContain("shorten your prompt or pick a shorter style");
    });

    // sc-13224: structured-caption models now DO apply the Style axis (the style is merged into the
    // caption's aesthetics), so the composed caption can push past the cap and the guard must fire.
    it("fires the budget error on a structured model when the injected caption exceeds the cap", () => {
      const overCaption = JSON.stringify({
        style_description: { aesthetics: "x".repeat(PROMPT_MAX_CHARS), photo: "f/2" },
        compositional_deconstruction: { background: "an alley", elements: [] },
      });
      expect([...overCaption].length).toBeGreaterThan(PROMPT_MAX_CHARS);
      const issues = imageGenerateValidation({
        ...whole,
        structuredActive: true,
        captionHasContent: true,
        styleActive: true,
        composedPrompt: overCaption,
      });
      const summary = summarize(issues);
      expect(summary.ready).toBe(false);
      expect(summary.surfaced.some((i) => i.kind === "error" && i.message.includes("/"))).toBe(true);
    });

    it("stays out of a structured model's gate when no style is active (empty composedPrompt)", () => {
      // styleless structured behavior is unchanged: no style selected → no composed prompt → no guard.
      const issues = imageGenerateValidation({
        ...whole,
        structuredActive: true,
        captionHasContent: true,
        styleActive: false,
        composedPrompt: "",
      });
      expect(summarize(issues).surfaced).toEqual([]);
    });
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
    promptBudgetOverages: [],
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

  it("blocks at 4001 Unicode scalars, allows 4000, and identifies every offending resolved item", () => {
    const composedPrompts = [
      "😀".repeat(PROMPT_MAX_CHARS),
      "a".repeat(PROMPT_MAX_CHARS + 1),
      "ok",
      "😀".repeat(PROMPT_MAX_CHARS + 2),
    ];
    const overages = batchPromptBudgetOverages(composedPrompts);
    expect(overages).toEqual([
      { item: 2, length: 4001, max: 4000, remaining: -1, over: true },
      { item: 4, length: 4002, max: 4000, remaining: -2, over: true },
    ]);

    const summary = summarize(imageBatchValidation({ ...whole, promptBudgetOverages: overages }));
    expect(summary.ready).toBe(false);
    expect(summary.surfaced[0].message).toBe(
      "Batch prompts 2 (4001/4000), 4 (4002/4000) exceed the character limit — shorten the prompt or pick a shorter style.",
    );
  });

  it("catches a batch item whose short raw prompt only exceeds the cap after style composition", () => {
    const rawPrompt = "a fox";
    const composedPrompt = composeStyledPrompt({
      styleText: "x".repeat(PROMPT_MAX_CHARS),
      userPrompt: rawPrompt,
    });
    expect([...rawPrompt].length).toBeLessThan(PROMPT_MAX_CHARS);
    expect([...composedPrompt].length).toBeGreaterThan(PROMPT_MAX_CHARS);

    const overages = batchPromptBudgetOverages([composedPrompt]);
    expect(overages).toHaveLength(1);
    expect(overages[0]).toMatchObject({ item: 1, length: [...composedPrompt].length, max: PROMPT_MAX_CHARS });
    expect(summarize(imageBatchValidation({ ...whole, promptBudgetOverages: overages })).ready).toBe(false);
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
