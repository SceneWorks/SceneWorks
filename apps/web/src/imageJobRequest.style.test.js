import { describe, expect, it } from "vitest";

import { buildImageJobRequest } from "./imageJobRequest.js";
import { composeStyledPrompt } from "./styleComposer.js";

// sc-13130: the Style Catalog composer is applied as the LAST wrap on the outgoing `prompt` inside
// buildImageJobRequest. These tests pin (a) identity when no style is selected, (b) exact
// composeStyledPrompt output when a style IS selected — using the preset-composed prompt the caller
// threads in as `promptToSend` for the `userPrompt` — plus the client-authoritative flag, and
// (c) that structured JSON-caption models never get the composer.

// A minimal but complete studio state for a plain text-to-image model. Only the fields the style
// fold touches matter here; the rest are defaults so the builder's other guards stay inert.
function baseState(overrides = {}) {
  return {
    promptToSend: "a fox in the snow",
    submitIntent: "a fox in the snow",
    sendStructured: false,
    submitCaption: null,
    submitBackend: null,
    resolutionOverride: null,
    resolution: "1024x1024",
    mode: "text_to_image",
    negativePrompt: "",
    model: "krea-2-turbo",
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
    editSecondPair: null,
    sourceAssetId: null,
    controlPreprocessSourceId: null,
    referenceAssetIds: [],
    fitMode: "crop",
    editInpaintCapable: false,
    referenceAssetId: null,
    supportsImg2img: false,
    img2imgReferenceAssetId: null,
    loras: [],
    upscaleEnabled: false,
    upscaleFactor: 2,
    upscaleEngine: "realesrgan",
    upscaleSoftness: 0.5,
    sampler: "default",
    scheduler: "default",
    schedulerShift: "",
    stepsOverride: "",
    guidanceOverride: "",
    guidanceMethod: "default",
    flashAttn: false,
    promptEnhance: false,
    enhancePrompt: false,
    precisionToggle: false,
    bf16Precision: false,
    showTierPicker: false,
    quantTier: "",
    showPidToggle: false,
    usePid: false,
    pidTarget: "",
    hideReferenceStrength: false,
    ipAdapterScale: 0,
    identityStructure: 0,
    controlnetScale: 0,
    variationStrength: 0,
    trueCfgScale: 0,
    img2imgStrength: 0,
    supportsTextStyle: false,
    textStyleGain: 0,
    viewAngles: [],
    viewAngle: "",
    faceRestore: false,
    controlActive: false,
    activeControlMode: null,
    controlPassthroughId: null,
    effectiveControlScale: 0,
    controlOverlayId: null,
    ...overrides,
  };
}

const STYLE_TEXT = "A gentle, hand-painted animation illustration style.";

describe("buildImageJobRequest — Style Catalog fold (sc-13130)", () => {
  it("no style selected → prompt is the untouched user prompt (identity)", () => {
    const req = buildImageJobRequest(baseState({ styleText: null }));
    expect(req.prompt).toBe("a fox in the snow");
    // No style applied and no preset fold → the flag stays undefined.
    expect(req.presetPromptResolvedClientSide).toBeUndefined();
  });

  it("empty/whitespace styleText is treated as pass-through", () => {
    expect(buildImageJobRequest(baseState({ styleText: "" })).prompt).toBe("a fox in the snow");
    expect(buildImageJobRequest(baseState({ styleText: "   " })).prompt).toBe("a fox in the snow");
    expect(buildImageJobRequest(baseState({ styleText: "   " })).presetPromptResolvedClientSide).toBeUndefined();
  });

  it("style selected → prompt equals composeStyledPrompt(styleText, promptToSend) and sets the client flag", () => {
    const state = baseState({ styleText: STYLE_TEXT });
    const req = buildImageJobRequest(state);
    expect(req.prompt).toBe(composeStyledPrompt({ styleText: STYLE_TEXT, userPrompt: "a fox in the snow" }));
    // Sanity: the composed prompt actually carries the Style:/Description: template.
    expect(req.prompt).toBe(`Style: ${STYLE_TEXT}\nDescription: a fox in the snow`);
    expect(req.presetPromptResolvedClientSide).toBe(true);
  });

  it("style composes on top of an already-preset-composed prompt (composer runs LAST)", () => {
    // The caller (ImageStudio fold) passes the preset-composed prompt as promptToSend; the builder
    // wraps THAT as the Description block — proving the ordering the story requires.
    const presetComposed = "cinematic, moody. a fox in the snow";
    const req = buildImageJobRequest(baseState({ styleText: STYLE_TEXT, promptToSend: presetComposed }));
    expect(req.prompt).toBe(composeStyledPrompt({ styleText: STYLE_TEXT, userPrompt: presetComposed }));
    expect(req.prompt).toBe(`Style: ${STYLE_TEXT}\nDescription: ${presetComposed}`);
  });

  it("preserves an existing presetPromptResolvedClientSide=true even with no style", () => {
    const req = buildImageJobRequest(baseState({ styleText: null, presetPromptResolvedClientSide: true }));
    expect(req.presetPromptResolvedClientSide).toBe(true);
  });

  it("structured JSON-caption models never get the composer (prompt passes through)", () => {
    const caption = '{"scene":"a fox"}';
    const req = buildImageJobRequest(
      baseState({ sendStructured: true, promptToSend: caption, styleText: STYLE_TEXT }),
    );
    expect(req.prompt).toBe(caption);
    expect(req.presetPromptResolvedClientSide).toBeUndefined();
  });
});
