import { describe, expect, it } from "vitest";
import {
  DEFAULT_IMG2IMG_STRENGTH,
  buildSimpleAudioRequest,
  buildSimpleImageRequest,
  buildSimpleVideoRequest,
  parseResolutionPair,
  referenceStrengthFor,
} from "./simpleJobs.js";
import { buildImageJobRequest } from "../imageJobRequest.js";
import { STYLE_GROUPS, styleTextForId } from "../data/styleCatalog.js";

const STYLE_GROUP_ID = STYLE_GROUPS[0].id;

describe("parseResolutionPair", () => {
  it("parses a WxH string and rejects anything that isn't a positive pair", () => {
    expect(parseResolutionPair("1344x768")).toEqual({ width: 1344, height: 768 });
    expect(parseResolutionPair("1024X1024")).toEqual({ width: 1024, height: 1024 });
    expect(parseResolutionPair("")).toBeNull();
    expect(parseResolutionPair("1024")).toBeNull();
    expect(parseResolutionPair("0x512")).toBeNull();
    expect(parseResolutionPair("widexhigh")).toBeNull();
    expect(parseResolutionPair(undefined)).toBeNull();
  });
});

describe("buildSimpleImageRequest", () => {
  const base = {
    prompt: "a lighthouse at golden hour",
    mode: "text_to_image",
    model: "z_image",
    resolution: "1344x768",
    count: 4,
    styleId: null,
    referenceAssetId: null,
  };

  it("emits the geometry, count and mode the studio shows", () => {
    const request = buildSimpleImageRequest(base);
    expect(request.mode).toBe("text_to_image");
    expect(request.model).toBe("z_image");
    expect(request.prompt).toBe("a lighthouse at golden hour");
    expect(request.count).toBe(4);
    expect(request.width).toBe(1344);
    expect(request.height).toBe(768);
    expect(request.advanced.resolution).toBe("1344x768");
  });

  it("refuses an unparseable resolution instead of posting a NaN geometry", () => {
    expect(buildSimpleImageRequest({ ...base, resolution: "not-a-size" })).toBeNull();
  });

  it("leaves `advanced` at just the resolution when no style and no tier are in play", () => {
    // This is the load-bearing claim: every advanced knob Simple hides must hit its
    // omit-when-default rule, so a Simple payload is byte-identical to an untouched
    // advanced run. A new always-emitted key would fail here.
    expect(Object.keys(buildSimpleImageRequest(base).advanced)).toEqual(["resolution"]);
  });

  it("is byte-identical to the advanced builder given the same visible settings", () => {
    // Guards the reuse claim directly: if IMAGE_DEFAULTS ever drifts from what the
    // advanced studio sends for an untouched control, this diverges.
    const simple = buildSimpleImageRequest(base);
    const advanced = buildImageJobRequest({
      promptToSend: base.prompt,
      submitIntent: base.prompt,
      sendStructured: false,
      submitCaption: null,
      submitBackend: null,
      resolutionOverride: null,
      resolution: "1344x768",
      mode: "text_to_image",
      negativePrompt: "",
      model: "z_image",
      count: 4,
      seed: "",
      posePayload: [],
      width: 1344,
      height: 768,
      recipePresetId: null,
      presetPromptResolvedClientSide: false,
      styleText: null,
      styleId: null,
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
      upscaleEngine: "",
      upscaleSoftness: 0,
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
      quantTier: "",
      showPidToggle: false,
      usePid: false,
      pidTarget: "4k",
      hideReferenceStrength: false,
      ipAdapterScale: 0,
      identityStructure: false,
      controlnetScale: 0,
      variationStrength: false,
      trueCfgScale: 0,
      img2imgStrength: 0,
      supportsTextStyle: false,
      textStyleGain: 1,
      viewAngles: null,
      viewAngle: "",
      faceRestore: false,
      controlActive: false,
      activeControlMode: "pose",
      controlPassthroughId: null,
      effectiveControlScale: 0,
      controlOverlayId: null,
      multiPhaseActive: false,
      phases: [],
    });
    expect(JSON.stringify(simple)).toBe(JSON.stringify(advanced));
  });

  it("composes a picked style into the prompt and records the id for replay", () => {
    const request = buildSimpleImageRequest({ ...base, styleId: STYLE_GROUP_ID });
    const styleText = styleTextForId(STYLE_GROUP_ID);
    expect(styleText).toBeTruthy();
    // The outgoing prompt is the COMPOSED string, not the raw one…
    expect(request.prompt).not.toBe(base.prompt);
    expect(request.prompt).toContain(styleText);
    // …the client is authoritative, so the server must skip its own fold…
    expect(request.presetPromptResolvedClientSide).toBe(true);
    // …and both halves needed to replay into the advanced Style picker are recorded.
    expect(request.advanced.styleId).toBe(STYLE_GROUP_ID);
    expect(request.advanced.stylePrompt).toBe(base.prompt);
  });

  it("routes an edit run's source image through sourceAssetId", () => {
    const request = buildSimpleImageRequest({
      ...base,
      mode: "edit_image",
      referenceAssetId: "asset-7",
    });
    expect(request.mode).toBe("edit_image");
    expect(request.sourceAssetId).toBe("asset-7");
    expect(request.fitMode).toBe("crop");
    // The edit path has no strength knob — the model's edit pipeline owns the conditioning.
    expect(request.advanced.strength).toBeUndefined();
    expect(request.referenceAssetId).toBeNull();
  });

  // The bug this suite missed on the first pass: a TEXT-mode reference was routed to
  // `sourceAssetId`, which buildImageJobRequest drops for non-edit modes — so the tile
  // armed, badged and toasted, and then nothing was sent. The reference has to ride the
  // img2img channel, which is also what makes `advanced.strength` appear.
  it("routes a TEXT-mode reference through img2img and sends the model's strength", () => {
    const request = buildSimpleImageRequest({
      ...base,
      referenceAssetId: "asset-9",
      supportsImg2img: true,
      img2imgStrength: 0.5,
    });
    expect(request.referenceAssetId).toBe("asset-9");
    expect(request.advanced.strength).toBe(0.5);
    // Never as the edit source — that field is edit-mode-only and would be discarded.
    expect(request.sourceAssetId).toBeNull();
  });

  it("honors a model-declared reference strength over the fallback", () => {
    const request = buildSimpleImageRequest({
      ...base,
      referenceAssetId: "asset-9",
      supportsImg2img: true,
      img2imgStrength: 0.35,
    });
    expect(request.advanced.strength).toBe(0.35);
  });

  it("drops a TEXT-mode reference on a model that can't do img2img", () => {
    const request = buildSimpleImageRequest({
      ...base,
      referenceAssetId: "asset-9",
      supportsImg2img: false,
    });
    expect(request.referenceAssetId).toBeNull();
    expect(request.sourceAssetId).toBeNull();
    // No strength for a reference that isn't being sent.
    expect(request.advanced.strength).toBeUndefined();
  });

  // Krea 2's edit lane REJECTS a run whose payload carries no `image_edit`-role LoRA
  // ("without it the source-image conditioning is inert"). Simple sent `loras: []`
  // unconditionally, so every Krea 2 edit failed in the worker.
  const KREA_EDIT_LORA = {
    id: "krea2_identity_edit",
    name: "Krea 2 Identity Edit",
    conditioningRole: "image_edit",
    installState: "installed",
  };

  it("auto-applies the model's managed edit LoRA on an edit run", () => {
    const request = buildSimpleImageRequest({
      ...base,
      mode: "edit_image",
      referenceAssetId: "asset-7",
      editLora: KREA_EDIT_LORA,
    });
    expect(request.loras).toHaveLength(1);
    expect(request.loras[0].id).toBe("krea2_identity_edit");
    // The ROLE is what the worker's edit-lane check reads — a serialized entry that lost it
    // fails exactly as an absent LoRA would.
    expect(request.loras[0].conditioningRole).toBe("image_edit");
  });

  it("does not attach the edit LoRA to a TEXT run", () => {
    const request = buildSimpleImageRequest({ ...base, editLora: KREA_EDIT_LORA });
    expect(request.loras).toEqual([]);
  });

  it("sends no LoRAs for an edit model that needs none", () => {
    const request = buildSimpleImageRequest({
      ...base,
      mode: "edit_image",
      referenceAssetId: "asset-7",
      editLora: null,
    });
    expect(request.loras).toEqual([]);
  });

  it("emits no img2img fields at all when nothing is armed", () => {
    const request = buildSimpleImageRequest({ ...base, supportsImg2img: true });
    expect(request.referenceAssetId).toBeNull();
    expect(request.advanced.strength).toBeUndefined();
  });

  it("emits the resolved quant tier, and marks it explicit only for a deliberate pick", () => {
    const derived = buildSimpleImageRequest({ ...base, quantTier: "q4", tierExplicit: false });
    expect(derived.advanced.mlxQuantize).toBe(4);
    expect(derived.advanced.mlxQuantizeExplicit).toBeUndefined();

    const sticky = buildSimpleImageRequest({ ...base, quantTier: "q4", tierExplicit: true });
    expect(sticky.advanced.mlxQuantizeExplicit).toBe(true);
  });
});

describe("referenceStrengthFor", () => {
  it("reads the model's declared default so a tuned catalog entry is honored", () => {
    expect(referenceStrengthFor({ ui: { img2imgStrength: { default: 0.35 } } })).toBe(0.35);
    // 0 is a legitimate strength and must not be treated as "unset".
    expect(referenceStrengthFor({ ui: { img2imgStrength: { default: 0 } } })).toBe(0);
  });

  it("falls back when the entry declares img2img but no strength config", () => {
    expect(referenceStrengthFor({ ui: { img2img: true } })).toBe(DEFAULT_IMG2IMG_STRENGTH);
    expect(referenceStrengthFor(null)).toBe(DEFAULT_IMG2IMG_STRENGTH);
    expect(referenceStrengthFor({ ui: { img2imgStrength: { default: "half" } } })).toBe(
      DEFAULT_IMG2IMG_STRENGTH,
    );
  });

  it("matches what every img2img entry in the shipped catalog declares", () => {
    // Pins the fallback to the catalog's actual value, so a manifest sweep that changed it
    // wouldn't leave this constant silently disagreeing with every model.
    expect(DEFAULT_IMG2IMG_STRENGTH).toBe(0.5);
  });
});

describe("buildSimpleVideoRequest", () => {
  const base = {
    prompt: "slow drone push-in over a misty forest",
    mode: "text_to_video",
    model: "ltx_2_3",
    resolution: "1280x720",
    duration: 5,
    styleId: null,
    sourceAssetId: null,
  };

  it("emits the geometry and duration the studio shows", () => {
    const request = buildSimpleVideoRequest(base);
    expect(request).toMatchObject({
      mode: "text_to_video",
      model: "ltx_2_3",
      duration: 5,
      width: 1280,
      height: 720,
      quality: "balanced",
      seed: null,
    });
    expect(request.advanced.resolution).toBe("1280x720");
  });

  // Simple exposes no frame-rate control, and the route resolves an absent fps from the
  // model's declared `defaults.fps` (sc-12347). Sending an invented rate is what the DTO
  // calls out as harmful — Wan 2.2 declares 24 but lists 16 first, so a `limits.fps[0]`
  // guess rendered it 33% slow.
  it("omits fps entirely so the route resolves the model's declared rate", () => {
    expect("fps" in buildSimpleVideoRequest(base)).toBe(false);
  });

  it("refuses an unparseable resolution", () => {
    expect(buildSimpleVideoRequest({ ...base, resolution: "720" })).toBeNull();
  });

  it("only carries a source asset in image_to_video", () => {
    expect(buildSimpleVideoRequest({ ...base, sourceAssetId: "asset-3" }).sourceAssetId).toBeNull();
    expect(
      buildSimpleVideoRequest({ ...base, mode: "image_to_video", sourceAssetId: "asset-3" })
        .sourceAssetId,
    ).toBe("asset-3");
  });

  it("wraps a picked style and records the raw prompt for replay", () => {
    const request = buildSimpleVideoRequest({ ...base, styleId: STYLE_GROUP_ID });
    expect(request.prompt).toContain(styleTextForId(STYLE_GROUP_ID));
    expect(request.presetPromptResolvedClientSide).toBe(true);
    expect(request.advanced.styleId).toBe(STYLE_GROUP_ID);
    expect(request.advanced.stylePrompt).toBe(base.prompt);
  });

  it("leaves a styleless run's advanced payload at just the resolution", () => {
    const request = buildSimpleVideoRequest(base);
    expect(Object.keys(request.advanced)).toEqual(["resolution"]);
    expect(request.presetPromptResolvedClientSide).toBeUndefined();
  });
});

describe("buildSimpleAudioRequest", () => {
  it("sends the base payload the audio route deserializes", () => {
    expect(
      buildSimpleAudioRequest({
        model: "acestep_v15_turbo",
        prompt: "  warm lo-fi beat  ",
        mode: "music",
        voice: "",
        durationSecs: 8,
      }),
    ).toEqual({
      model: "acestep_v15_turbo",
      prompt: "warm lo-fi beat",
      targetDurationSecs: 8,
      seed: null,
    });
  });

  it("carries a voice only in Speech", () => {
    const speech = buildSimpleAudioRequest({
      model: "kokoro_82m",
      prompt: "hello",
      mode: "speech",
      voice: "af_heart",
      durationSecs: 8,
    });
    expect(speech.voice).toBe("af_heart");
    const music = buildSimpleAudioRequest({
      model: "acestep_v15_turbo",
      prompt: "hello",
      mode: "music",
      voice: "af_heart",
      durationSecs: 8,
    });
    expect(music.voice).toBeUndefined();
  });

  it("omits the target duration when the model declares no length cap", () => {
    const request = buildSimpleAudioRequest({
      model: "moss_sfx_v2",
      prompt: "thunder",
      mode: "sfx",
      voice: "",
      durationSecs: null,
    });
    expect(request.targetDurationSecs).toBeUndefined();
  });
});
