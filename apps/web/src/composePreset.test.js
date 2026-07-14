import { describe, expect, it } from "vitest";

import { composePreset } from "./presetUtils.js";

// epic 11949: the general-preset stack composes FLAT (fragments, not nested wraps). These pin
// the exact concatenation the studio preview shows and a client-authoritative job sends.
describe("composePreset", () => {
  const general = (id, prompt = {}, defaults = {}) => ({ id, kind: "general", prompt, defaults });

  it("stacks append-only fragments in selection order (the camera/lens/film case)", () => {
    const { prompt } = composePreset({
      generalStack: [
        general("camera", { suffix: "shot on Arri Alexa" }),
        general("lens", { suffix: "85mm f/1.4" }),
        general("film", { suffix: "Kodak Portra 400" }),
      ],
      userText: "a fox in the snow",
    });
    expect(prompt).toBe("a fox in the snow, shot on Arri Alexa, 85mm f/1.4, Kodak Portra 400");
  });

  it("prepends a leading-style fragment before the user's prompt", () => {
    const { prompt } = composePreset({
      generalStack: [general("style", { prefix: "cinematic still" })],
      userText: "a fox",
    });
    expect(prompt).toBe("cinematic still, a fox");
  });

  it("brackets the base model preset's prefix/suffix outermost, generals inside", () => {
    const { prompt } = composePreset({
      base: { id: "m", prompt: { prefix: "masterpiece", suffix: "best quality" } },
      generalStack: [general("film", { prefix: "moody", suffix: "grainy" })],
      userText: "a fox",
    });
    // base.prefix, gen.prefix, USER, gen.suffix, base.suffix
    expect(prompt).toBe("masterpiece, moody, a fox, grainy, best quality");
  });

  it("drops the user slot when the prompt is empty", () => {
    const { prompt } = composePreset({
      generalStack: [general("film", { prefix: "cinematic", suffix: "Kodak Portra 400" })],
      userText: "   ",
    });
    expect(prompt).toBe("cinematic, Kodak Portra 400");
  });

  it("concatenates the user's negative with every stacked general's negative", () => {
    const { negativePrompt } = composePreset({
      generalStack: [
        general("a", {}, { negativePrompt: "blurry" }),
        general("b", {}, { negativePrompt: "watermark" }),
      ],
      userText: "x",
      userNegative: "lowres",
    });
    expect(negativePrompt).toBe("lowres, blurry, watermark");
  });

  it("resolves the last stacked general's aspect to the nearest supported resolution", () => {
    const { aspect, resolution } = composePreset({
      generalStack: [general("a", {}, { aspect: "1:1" }), general("b", {}, { aspect: "16:9" })],
      userText: "x",
      resolutionOptions: ["1024x1024", "1280x720", "720x1280"],
    });
    expect(aspect).toBe("16:9");
    expect(resolution).toBe("1280x720");
  });

  it("lets a base model preset's resolution win over a stacked general's aspect", () => {
    const { resolution } = composePreset({
      base: { id: "m", defaults: { resolution: "1024x1024" } },
      generalStack: [general("b", {}, { aspect: "16:9" })],
      userText: "x",
      resolutionOptions: ["1024x1024", "1280x720"],
    });
    expect(resolution).toBe("1024x1024");
  });

  it("takes count from the base preset first, else the last stacked general", () => {
    expect(
      composePreset({
        base: { id: "m", defaults: { count: 4 } },
        generalStack: [general("b", {}, { count: 2 })],
      }).count,
    ).toBe(4);
    expect(
      composePreset({
        generalStack: [general("a", {}, { count: 3 }), general("b", {}, { count: 2 })],
      }).count,
    ).toBe(2);
    // Nothing sets it → null so the studio keeps its own value.
    expect(composePreset({ generalStack: [general("a")], userText: "x" }).count).toBeNull();
  });

  it("returns the bare prompt with no presets active", () => {
    expect(composePreset({ userText: "just me" })).toEqual({
      prompt: "just me",
      negativePrompt: "",
      aspect: null,
      resolution: null,
      count: null,
    });
  });
});
