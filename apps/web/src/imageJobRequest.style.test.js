import { describe, expect, it } from "vitest";

import { buildImageJobRequest } from "./imageJobRequest.js";
import { composeStyledPrompt } from "./styleComposer.js";
import { styleTextForId } from "./data/styleCatalog.js";
import {
  injectStyleIntoCaption,
  parseCaption,
  serializeCaption,
} from "./ideogramCaption.js";

// sc-13130 / sc-13224: the Style Catalog is applied as the LAST wrap on the outgoing `prompt` inside
// buildImageJobRequest. These tests pin (a) identity when no style is selected, (b) exact
// composeStyledPrompt output for PROSE models when a style IS selected — using the preset-composed
// prompt the caller threads in as `promptToSend` for the `userPrompt` — plus the client-authoritative
// flag, and (c) that STRUCTURED JSON-caption models get the caption injection (into
// style_description.aesthetics), never the prose composer — and only when the prompt is actually a
// valid caption that gets transformed (a non-caption structured prompt passes through with no flag,
// so the server's own fold can still handle it and the style is never silently dropped).

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

  it("structured model with a VALID caption → style injected into aesthetics, prose composer NOT used, flag set", () => {
    // A structurally-valid caption (has the required compositional_deconstruction section).
    const caption = '{"compositional_deconstruction": {"background": "a snowy forest", "elements": []}}';
    const req = buildImageJobRequest(
      baseState({
        sendStructured: true,
        promptToSend: caption,
        styleText: STYLE_TEXT,
        styleId: "ghibli-style",
      }),
    );

    // The outgoing prompt is the caption with the style merged into style_description.aesthetics —
    // exactly what injectStyleIntoCaption + serializeCaption produce (NOT the prose Style:/Description:
    // wrap, which would be wrong for a JSON-caption model).
    const expected = serializeCaption(
      injectStyleIntoCaption(parseCaption(caption).caption, STYLE_TEXT),
    );
    expect(req.prompt).toBe(expected);
    expect(req.prompt).not.toBe(caption); // the caption was actually transformed
    expect(req.prompt.startsWith("Style:")).toBe(false); // no prose composer
    expect(req.prompt).toContain(STYLE_TEXT); // style folded into aesthetics

    // An injection actually happened, so the client is authoritative and records the picker id.
    expect(req.presetPromptResolvedClientSide).toBe(true);
    expect(req.advanced.styleId).toBe("ghibli-style");
    // For structured injection the raw prompt lives in the caption blob, so stylePrompt is "".
    expect(req.advanced.stylePrompt).toBe("");
  });

  it("structured model with a NON-caption prompt → passed through unchanged, no flag/styleId (server can still fold)", () => {
    // `{"scene":"a fox"}` is JSON but not a caption (no compositional_deconstruction). The injection
    // no-ops, so the prompt must be left untouched AND the client-authoritative flag/styleId must NOT
    // be set — otherwise the server would skip its own fold and the style would be silently dropped.
    const notACaption = '{"scene":"a fox"}';
    const req = buildImageJobRequest(
      baseState({
        sendStructured: true,
        promptToSend: notACaption,
        styleText: STYLE_TEXT,
        styleId: "ghibli-style",
      }),
    );
    expect(req.prompt).toBe(notACaption);
    expect(req.presetPromptResolvedClientSide).toBeUndefined();
    expect(req.advanced.styleId).toBeUndefined();
  });
});

// sc-13132: the recipe must reproduce the SAME style composition on replay. The picked style id and
// the RAW pre-style prompt ride `advanced` (→ the asset's rawAdapterSettings, cloned verbatim by the
// worker, no backend change). These tests pin (a) that both are recorded only when a style is
// applied, (b) that the id round-trips for a sub-style id AND a group id, and (c) the load-bearing
// guarantee: replaying the recorded fields recomposes the IDENTICAL prompt with no double-wrap.
describe("buildImageJobRequest — Style Catalog recipe round-trip (sc-13132)", () => {
  const SUBSTYLE_ID = "ghibli-style"; // a sub-style id in styles.json
  const GROUP_ID = "anime-style"; // a group id (the group's generic "overall" style, sc-13171)

  it("no style selected → advanced records neither styleId nor stylePrompt", () => {
    const req = buildImageJobRequest(baseState({ styleText: null, styleId: null }));
    expect(req.advanced.styleId).toBeUndefined();
    expect(req.advanced.stylePrompt).toBeUndefined();
  });

  it("style applied → advanced records the picked styleId and the RAW pre-style prompt", () => {
    const req = buildImageJobRequest(
      baseState({ styleText: STYLE_TEXT, styleId: SUBSTYLE_ID, promptToSend: "a fox in the snow" }),
    );
    expect(req.advanced.styleId).toBe(SUBSTYLE_ID);
    // The stored prompt is the RAW (pre-style) prompt, NOT the composed top-level prompt.
    expect(req.advanced.stylePrompt).toBe("a fox in the snow");
    expect(req.advanced.stylePrompt).not.toBe(req.prompt);
  });

  it("a style with no user prose still round-trips exactly (stylePrompt = \"\")", () => {
    const req = buildImageJobRequest(
      baseState({ styleText: STYLE_TEXT, styleId: SUBSTYLE_ID, promptToSend: "" }),
    );
    expect(req.advanced.styleId).toBe(SUBSTYLE_ID);
    expect(req.advanced.stylePrompt).toBe("");
  });

  // Simulate the studio: resolve the picked id → styleText (ImageStudio's styleTextForId bridge),
  // build the job, then REPLAY by reading the recorded advanced.{styleId, stylePrompt} exactly as
  // the rehydrate effect does, and rebuild. Asserts the composed prompt is bit-identical and never
  // double-wrapped, for BOTH a sub-style id and a group id.
  for (const styleId of [SUBSTYLE_ID, GROUP_ID]) {
    it(`replay of a recorded recipe (${styleId}) recomposes the identical prompt — no double-wrap`, () => {
      const RAW = "a fox in the snow";
      const styleText = styleTextForId(styleId);
      expect(typeof styleText).toBe("string"); // guards a stale test id

      // Original generate.
      const original = buildImageJobRequest(
        baseState({ styleText, styleId, promptToSend: RAW }),
      );
      // The recipe records these two facts (advanced → rawAdapterSettings).
      const recorded = { styleId: original.advanced.styleId, stylePrompt: original.advanced.stylePrompt };
      expect(recorded.styleId).toBe(styleId);
      expect(recorded.stylePrompt).toBe(RAW);

      // Replay: the rehydrate effect re-selects the picker (recorded.styleId) and seeds the box with
      // the RAW prompt (recorded.stylePrompt); the next submit resolves styleText from the id again.
      const replay = buildImageJobRequest(
        baseState({ styleText: styleTextForId(recorded.styleId), styleId: recorded.styleId, promptToSend: recorded.stylePrompt }),
      );
      expect(replay.prompt).toBe(original.prompt);
      // Exactly one Style: block — the composer did not wrap an already-composed prompt.
      expect(replay.prompt.match(/^Style:/gm)?.length ?? 0).toBe(1);
    });
  }

  it("proves the trap the raw-prompt storage avoids: recomposing over the COMPOSED prompt double-wraps", () => {
    const RAW = "a fox in the snow";
    const composed = buildImageJobRequest(
      baseState({ styleText: STYLE_TEXT, styleId: SUBSTYLE_ID, promptToSend: RAW }),
    ).prompt;
    // If replay had (wrongly) seeded the box with the COMPOSED prompt while keeping the style
    // applied, the composer would merge a second copy of the style into the Style: block.
    const doubleWrapped = buildImageJobRequest(
      baseState({ styleText: STYLE_TEXT, styleId: SUBSTYLE_ID, promptToSend: composed }),
    ).prompt;
    expect(doubleWrapped).not.toBe(composed);
    expect(doubleWrapped.startsWith(`Style: ${STYLE_TEXT}, ${STYLE_TEXT}`)).toBe(true);
  });
});
