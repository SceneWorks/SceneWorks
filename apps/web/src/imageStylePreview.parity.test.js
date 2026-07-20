import { describe, expect, it } from "vitest";

import { buildImageJobRequest } from "./imageJobRequest.js";
import { composePreset } from "./presetUtils.js";
import { composeStyledPrompt } from "./styleComposer.js";
import { STYLE_GROUPS, styleTextForId } from "./data/styleCatalog.js";

// sc-13131 — CORE DoD guard: the live Style preview must equal the submitted `prompt` BYTE-FOR-BYTE.
//
// In ImageStudio the preview value is `buildJobRequest({ promptToSend: prompt }).prompt` — the SAME
// function the single Generate submit calls — so the preview cannot drift from the payload by
// construction. `buildJobRequest` folds the general-preset stack into the user text FIRST (via
// composePreset), then hands that preset-folded prompt to `buildImageJobRequest`, which applies the
// style wrap LAST (composeStyledPrompt). This suite pins that final `buildImageJobRequest` composition
// step — the one both the preview and the payload share — against the composer directly, so a change
// that made the payload compose differently from `composeStyledPrompt` (i.e. from the preview) fails.

// A representative studio state for a plain free-text (non-structured) model. Only the fields
// buildImageJobRequest reads need to be present; arrays are empty and edit/upscale branches are off.
function baseState(overrides = {}) {
  return {
    promptToSend: "",
    submitIntent: "",
    sendStructured: false,
    submitCaption: null,
    submitBackend: null,
    resolutionOverride: null,
    resolution: "1024x1024",
    mode: "text_to_image",
    negativePrompt: "",
    model: { id: "z_image_turbo", family: "z-image" },
    count: 1,
    seed: "",
    posePayload: [],
    width: 1024,
    height: 1024,
    recipePresetId: null,
    presetPromptResolvedClientSide: false,
    styleText: null,
    characterId: null,
    characterLookId: null,
    multiReference: false,
    editSecondPair: undefined,
    sourceAssetId: null,
    controlPreprocessSourceId: null,
    referenceAssetIds: [],
    fitMode: undefined,
    editInpaintCapable: false,
    referenceAssetId: null,
    supportsImg2img: false,
    img2imgReferenceAssetId: null,
    loras: [],
    upscaleEnabled: false,
    viewAngles: [],
    posePayloadLength: 0,
    ...overrides,
  };
}

const styleId = STYLE_GROUPS[0].styles[0].id; // "ghibli-style"
const styleText = styleTextForId(styleId);

describe("Style preview ↔ payload prompt parity (sc-13131)", () => {
  // The exact recipe ImageStudio uses to derive the preview / payload prompt from the raw prompt:
  //   1. fold the preset stack into the user text (composePreset) — FIRST,
  //   2. wrap the preset-folded prompt in the selected style (buildImageJobRequest → composeStyledPrompt) — LAST.
  // Returns { payloadPrompt } exactly as the submit path produces it.
  function payloadPromptFor({ prompt, stack = [] }) {
    const foldPrompt = stack.length > 0;
    const promptToSend = foldPrompt ? composePreset({ generalStack: stack, userText: prompt }).prompt : prompt;
    return buildImageJobRequest(baseState({ promptToSend, styleText })).prompt;
  }

  it("plain prompt: payload prompt equals the composer's Subject/Style block", () => {
    const prompt = "a fox in the snow";
    const payload = payloadPromptFor({ prompt });
    // The payload prompt IS what composeStyledPrompt produces for the same inputs — this is the
    // string the preview renders. If the builder ever stopped using composeStyledPrompt, this breaks.
    expect(payload).toBe(composeStyledPrompt({ styleText, userPrompt: prompt }));
    expect(payload).toBe(`Subject: ${prompt}\nStyle: ${styleText}`);
  });

  it("MERGE case: the user's own Style: line merges into the catalog style, visibly", () => {
    const prompt = "Style: neon rimlight\na fox in the snow";
    const payload = payloadPromptFor({ prompt });
    expect(payload).toBe(composeStyledPrompt({ styleText, userPrompt: prompt }));
    // The catalog style leads; the user's own style words follow after ", " in the SAME Style block.
    expect(payload).toBe(`Subject: a fox in the snow\nStyle: ${styleText}, neon rimlight`);
    expect(payload).toContain(", neon rimlight");
  });

  it("sibling directive stays a top-level sibling, not demoted under Subject", () => {
    const prompt = "Lighting: soft window light\na fox in the snow";
    const payload = payloadPromptFor({ prompt });
    expect(payload).toBe(composeStyledPrompt({ styleText, userPrompt: prompt }));
    expect(payload).toBe(`Subject: a fox in the snow\nStyle: ${styleText}\nLighting: soft window light`);
  });

  it("active preset: preset fragments fold into the Subject FIRST, style wraps LAST", () => {
    const stack = [{ id: "cine", name: "Cinematic", prompt: { prefix: "cinematic", suffix: "film grain" } }];
    const prompt = "a fox in the snow";
    const presetFolded = composePreset({ generalStack: stack, userText: prompt }).prompt;
    const payload = payloadPromptFor({ prompt, stack });
    // The preset-folded prompt is what lands inside Subject; the style still wraps it last.
    expect(payload).toBe(composeStyledPrompt({ styleText, userPrompt: presetFolded }));
    expect(payload).toBe(`Subject: cinematic, a fox in the snow, film grain\nStyle: ${styleText}`);
    // Discriminates a wrong composition order (style folded before the preset, or preset lost).
    expect(payload).toContain("Subject: cinematic, a fox in the snow, film grain");
  });

  it("no style selected: payload prompt is the plain prompt, unwrapped", () => {
    const prompt = "a fox in the snow";
    const payload = buildImageJobRequest(baseState({ promptToSend: prompt, styleText: null })).prompt;
    expect(payload).toBe(prompt);
    expect(payload).not.toContain("Style:");
  });
});
