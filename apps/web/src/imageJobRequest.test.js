import { describe, expect, it } from "vitest";

import { buildImageJobRequest } from "./imageJobRequest.js";
import { serializeCaption } from "./ideogramCaption.js";

// sc-11219 (F-031): the batch job-request builder used to be a hand-copied twin of the single
// Generate payload that had DRIFTED — the batch copy dropped the top-level `referenceAssetId`
// img2img branch and omitted `pidTarget` + the `supportsImg2img`/`img2imgReferenceAssetId`/
// `img2imgStrength` trio from `advanced`. So a batch on an img2img-capable model (Krea 2 Turbo)
// silently ignored the reference image and a PiD "2K" batch rendered at the 4K default. Both paths
// now go through buildImageJobRequest; these tests pin that a batch-style call (with the per-item
// prompt / caption / [WxH] override) produces the same img2img + PiD payload as single Generate.

// A studio state for an img2img-capable text-to-image model with a "Start from an image"
// reference picked and the PiD decoder set to the 2K tier — exactly the settings the drifted
// batch copy used to silently discard.
function img2imgPidState(overrides = {}) {
  return {
    // Prompt / resolution overrides (the one legitimate single-vs-batch difference).
    promptToSend: "a fox",
    submitIntent: "a fox",
    sendStructured: false,
    submitCaption: null,
    submitBackend: null,
    resolutionOverride: null,
    resolution: "1024x1024",
    // Shared studio settings.
    mode: "text_to_image",
    negativePrompt: "",
    model: "krea-2-turbo",
    count: 1,
    seed: "",
    posePayload: [],
    width: 1024,
    height: 1024,
    recipePresetId: null,
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
    // The img2img trio: an img2img-capable model with a reference chosen in the shared panel.
    supportsImg2img: true,
    img2imgReferenceAssetId: "start-image-asset",
    img2imgStrength: 0.42,
    loras: [],
    upscaleEnabled: false,
    upscaleFactor: 2,
    upscaleEngine: "realesrgan",
    upscaleSoftness: 0.5,
    // Advanced knobs — everything default except the PiD 2K tier.
    sampler: "default",
    scheduler: "default",
    schedulerShift: "",
    stepsOverride: "",
    guidanceOverride: "",
    guidanceMethod: "cfg",
    flashAttn: true,
    promptEnhance: false,
    enhancePrompt: false,
    precisionToggle: false,
    bf16Precision: false,
    showTierPicker: false,
    quantTier: "default",
    showPidToggle: true,
    usePid: true,
    pidTarget: "2k",
    hideReferenceStrength: false,
    ipAdapterScale: 0.8,
    identityStructure: false,
    controlnetScale: 0.5,
    variationStrength: false,
    trueCfgScale: 4,
    viewAngles: false,
    viewAngle: "",
    faceRestore: false,
    controlActive: false,
    activeControlMode: null,
    controlPassthroughId: null,
    effectiveControlScale: 0.7,
    controlOverlayId: null,
    ...overrides,
  };
}

describe("buildImageJobRequest", () => {
  it("routes the img2img reference to the top-level referenceAssetId on an img2img model", () => {
    // The branch the drifted batch copy dropped (it hardcoded character-only referenceAssetId).
    const request = buildImageJobRequest(img2imgPidState());
    expect(request.referenceAssetId).toBe("start-image-asset");
  });

  it("emits advanced.strength for the img2img trio (dropped by the drifted batch copy)", () => {
    const request = buildImageJobRequest(img2imgPidState());
    expect(request.advanced.strength).toBe(0.42);
  });

  it("emits advanced.pidTarget:'2k' when the PiD 2K tier is selected (dropped by the batch copy)", () => {
    const request = buildImageJobRequest(img2imgPidState());
    expect(request.advanced.usePid).toBe(true);
    expect(request.advanced.pidTarget).toBe("2k");
  });

  it("produces an identical img2img + PiD payload for a batch-style call as for single Generate", () => {
    // Single Generate: no per-prompt resolution override.
    const single = buildImageJobRequest(img2imgPidState());
    // Batch: a per-prompt prompt + [WxH] directive rides `overrides`, everything else identical.
    const batch = buildImageJobRequest(
      img2imgPidState({
        promptToSend: "a fox in the snow",
        submitIntent: "a fox in the snow",
        resolutionOverride: { width: 1280, height: 720 },
      }),
    );

    // The img2img/PiD-relevant fields the drift affected must match single exactly.
    expect(batch.referenceAssetId).toBe(single.referenceAssetId);
    expect(batch.advanced.strength).toBe(single.advanced.strength);
    expect(batch.advanced.usePid).toBe(single.advanced.usePid);
    expect(batch.advanced.pidTarget).toBe(single.advanced.pidTarget);

    // The ONLY intended differences are the prompt + the per-prompt resolution.
    expect(batch.prompt).toBe("a fox in the snow");
    expect(batch.width).toBe(1280);
    expect(batch.height).toBe(720);
    expect(batch.advanced.resolution).toBe("1280x720");

    // Everything else is byte-identical to single Generate.
    const stripVariant = (req) => {
      const { prompt: _p, width: _w, height: _h, advanced, ...rest } = req;
      const { resolution: _r, ...advRest } = advanced;
      return { ...rest, advanced: advRest };
    };
    expect(stripVariant(batch)).toEqual(stripVariant(single));
  });

  it("keeps the top-level referenceAssetId as the character reference in character_image mode", () => {
    // Regression guard: the img2img branch must not steal the character-reference slot.
    const request = buildImageJobRequest(
      img2imgPidState({
        mode: "character_image",
        referenceAssetId: "character-ref",
        supportsImg2img: false,
        img2imgReferenceAssetId: null,
      }),
    );
    expect(request.referenceAssetId).toBe("character-ref");
    expect(request.advanced).not.toHaveProperty("strength");
  });

  // Krea 2 multi-phase denoise round-trip (epic 13879 S5, sc-13885): a krea_2_raw t2i job with the
  // editor active carries the serialized `advanced.phases` verbatim, and the phase LoRA indices
  // point into the request's OWN `loras` array — the SAME order the worker resolves to
  // `LoadSpec::adapters`. The canonical S4 example: 4 steps Raw CFG-on base-only, then 4 steps Raw +
  // the turbo LoRA (index 1) CFG off.
  it("round-trips advanced.phases only when multi-phase is active, indexing the request's own loras", () => {
    const loras = [
      { id: "style-lora", name: "Watercolor" },
      { id: "krea2_turbo_accel", name: "Krea Turbo" },
    ];
    const canonicalPhases = [
      { steps: 4, guidance: 3.5, loras: [] },
      { steps: 4, guidance: 0, loras: [{ index: 1 }] },
    ];
    const state = img2imgPidState({
      model: "krea_2_raw",
      supportsImg2img: false,
      img2imgReferenceAssetId: null,
      showPidToggle: false,
      usePid: false,
      loras,
      multiPhaseActive: true,
      phases: canonicalPhases,
    });

    const request = buildImageJobRequest(state);
    // The phase list rides advanced.phases unchanged (the worker parse shape).
    expect(request.advanced.phases).toEqual(canonicalPhases);
    // Index 1 in the phase's loras resolves to the turbo LoRA in the request's own loras array.
    expect(request.loras[request.advanced.phases[1].loras[0].index].id).toBe("krea2_turbo_accel");

    // Inactive editor → NO advanced.phases (a single-phase Raw job is byte-for-byte unchanged).
    expect(
      buildImageJobRequest({ ...state, multiPhaseActive: false }).advanced,
    ).not.toHaveProperty("phases");
  });
});

// sc-13224: the Style axis applied to a structured JSON-caption model (Ideogram 4). The style is
// merged into `style_description.aesthetics` and the caption re-serialized — NOT wrapped in prose.
describe("buildImageJobRequest — Style axis on a structured caption model", () => {
  const CAPTION = {
    style_description: { aesthetics: "moody", lighting: "low key", photo: "f/1.8" },
    compositional_deconstruction: { background: "an alley", elements: [] },
  };

  function structuredStyleState(overrides = {}) {
    return img2imgPidState({
      model: "ideogram_4",
      supportsImg2img: false,
      img2imgReferenceAssetId: null,
      sendStructured: true,
      submitCaption: CAPTION,
      submitBackend: null,
      promptToSend: serializeCaption(CAPTION),
      submitIntent: "an alley",
      styleId: "cinematic-style",
      styleText: "cinematic watercolor",
      ...overrides,
    });
  }

  it("merges the style into style_description.aesthetics and re-serializes (user words first)", () => {
    const request = buildImageJobRequest(structuredStyleState());
    expect(request.prompt).toBe(
      serializeCaption({
        style_description: {
          aesthetics: "moody. cinematic watercolor",
          lighting: "low key",
          photo: "f/1.8",
        },
        compositional_deconstruction: { background: "an alley", elements: [] },
      }),
    );
  });

  it("sets presetPromptResolvedClientSide so the server does not double-inject", () => {
    const request = buildImageJobRequest(structuredStyleState());
    expect(request.presetPromptResolvedClientSide).toBe(true);
  });

  it("records styleId (and an empty stylePrompt) in advanced for replay", () => {
    const request = buildImageJobRequest(structuredStyleState());
    expect(request.advanced.styleId).toBe("cinematic-style");
    expect(request.advanced.stylePrompt).toBe("");
    // The recipe caption stays the PRE-injection caption so replay re-injects once, not twice.
    expect(request.advanced.structuredPrompt.caption).toEqual(CAPTION);
  });

  it("sends the caption unchanged when no style is selected", () => {
    const request = buildImageJobRequest(structuredStyleState({ styleId: null, styleText: "" }));
    expect(request.prompt).toBe(serializeCaption(CAPTION));
    expect(request.advanced).not.toHaveProperty("styleId");
    expect(request.presetPromptResolvedClientSide).toBeUndefined();
  });

  it("leaves the prose (non-structured) style path unchanged", () => {
    const request = buildImageJobRequest(
      img2imgPidState({
        model: "krea-2-turbo",
        promptToSend: "a fox in the snow",
        styleId: "cinematic-style",
        styleText: "cinematic watercolor",
      }),
    );
    expect(request.prompt).toBe("Subject: a fox in the snow\nStyle: cinematic watercolor");
    expect(request.presetPromptResolvedClientSide).toBe(true);
  });
});
