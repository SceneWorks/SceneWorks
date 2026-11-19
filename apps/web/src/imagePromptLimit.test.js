import { describe, expect, it } from "vitest";
import {
  batchClipNotice,
  clipPromptChars,
  fitPromptToLimit,
  imagePromptCharLimit,
  MAX_IMAGE_PROMPT_CHARS,
  MAX_PROMPT_CHARS,
  promptCharCount,
} from "./imagePromptLimit.js";
import { composeStyledPrompt } from "./styleComposer.js";

describe("imagePromptCharLimit", () => {
  it("follows the declared token window, between 4,000 and 16,384 characters", () => {
    expect(imagePromptCharLimit({ image: { maxPromptTokens: 4096 } })).toBe(16384);
    expect(imagePromptCharLimit({ image: { maxPromptTokens: 1024 } })).toBe(4096);
    expect(imagePromptCharLimit({ image: { maxPromptTokens: 128 } })).toBe(MAX_PROMPT_CHARS);
    expect(imagePromptCharLimit({ image: { maxPromptTokens: 100000 } })).toBe(MAX_IMAGE_PROMPT_CHARS);
    expect(imagePromptCharLimit({})).toBe(MAX_PROMPT_CHARS);
    expect(imagePromptCharLimit(null)).toBe(MAX_PROMPT_CHARS);
    expect(imagePromptCharLimit({ image: { maxPromptTokens: "4096" } })).toBe(MAX_PROMPT_CHARS);
  });
});

describe("fitPromptToLimit", () => {
  it("leaves a prompt that fits untouched", () => {
    expect(fitPromptToLimit("a fox", 10)).toEqual({ prompt: "a fox", clipped: false, composedLength: 5 });
  });

  it("cuts the user's text by the composed overflow, keeping the wrapper whole", () => {
    const compose = (text) => `Subject: ${text}\nStyle: watercolor`;
    const prompt = "x".repeat(50);
    const limit = 40;
    const fitted = fitPromptToLimit(prompt, limit, compose);
    expect(fitted.clipped).toBe(true);
    expect(promptCharCount(compose(fitted.prompt))).toBe(limit);
    expect(compose(fitted.prompt).endsWith("Style: watercolor")).toBe(true);
  });

  // Cutting into a recognised `Style:` directive changes how the rest is wrapped, so one cut by
  // the first overflow can still compose past the limit; the fit recomposes until it fits.
  it("recomposes after cutting through a recognised directive", () => {
    const compose = (text) => composeStyledPrompt({ styleText: "soft watercolor, warm light", userPrompt: text });
    const prompt = `${"x".repeat(200)}\nStyle: y`;
    const full = promptCharCount(compose(prompt));
    for (let overflow = 1; overflow <= 12; overflow += 1) {
      const limit = full - overflow;
      const fitted = fitPromptToLimit(prompt, limit, compose);
      expect(fitted.clipped).toBe(true);
      expect(promptCharCount(compose(fitted.prompt))).toBeLessThanOrEqual(limit);
    }
  });

  it("counts Unicode scalars like the server", () => {
    expect(clipPromptChars("😀😀😀", 2)).toBe("😀😀");
    expect(fitPromptToLimit("😀".repeat(5), 4).prompt).toBe("😀".repeat(4));
  });
});

describe("batchClipNotice", () => {
  it("names the items and the model, and is silent with none", () => {
    expect(batchClipNotice({ items: [], limit: 4000 })).toBeNull();
    expect(batchClipNotice({ items: [28], limit: 16384, modelName: "MiniMax-H3-Image" })).toBe(
      "Batch item 28 is longer than MiniMax-H3-Image accepts (16,384 characters). It will run with the end cut off.",
    );
    expect(batchClipNotice({ items: [1, 2, 3, 4, 5, 6, 7, 8], total: 10, limit: 4000 })).toContain(
      "1, 2, 3, 4, 5, 6, 7, 8 and 2 more are longer than this model accepts",
    );
  });
});
