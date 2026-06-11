import { describe, expect, it } from "vitest";

import { buildJoyCaptionPrompt, joyCaptionPromptMap } from "./joyCaptionPrompts.js";

// sc-4199: buildJoyCaptionPrompt was pure logic buried in the 2.2k-line
// TrainingStudio screen. Now extracted, it's directly testable.
describe("buildJoyCaptionPrompt", () => {
  it("selects the {length} template for a descriptive length", () => {
    const prompt = buildJoyCaptionPrompt({ captionType: "Descriptive", captionLength: "long", extraOptions: [] });
    expect(prompt).toBe("Write a long detailed description for this image.");
  });

  it("selects the unbounded template for length 'any'", () => {
    const prompt = buildJoyCaptionPrompt({ captionType: "Descriptive", captionLength: "any", extraOptions: [] });
    expect(prompt).toBe("Write a detailed description for this image.");
  });

  it("selects the word-count template for a numeric length and substitutes {word_count}", () => {
    const prompt = buildJoyCaptionPrompt({ captionType: "Descriptive", captionLength: "40", extraOptions: [] });
    expect(prompt).toBe("Write a detailed description for this image in 40 words or less.");
  });

  it("appends extra options and substitutes {name}", () => {
    const prompt = buildJoyCaptionPrompt({
      captionType: "Descriptive",
      captionLength: "any",
      nameInput: "Ada",
      extraOptions: ["Refer to the subject as {name}."],
    });
    expect(prompt).toBe("Write a detailed description for this image. Refer to the subject as Ada.");
  });

  it("falls back to the Descriptive corpus for an unknown caption type", () => {
    const prompt = buildJoyCaptionPrompt({ captionType: "Nonexistent", captionLength: "any", extraOptions: [] });
    expect(prompt).toBe(joyCaptionPromptMap.Descriptive[0]);
  });
});
