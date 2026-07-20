import { describe, expect, it } from "vitest";
import { DEFAULT_MAC_CAPABILITIES } from "./macGating.js";
import {
  AUDIO_MODES,
  angleModelUsable,
  audioModelServesMode,
  audioModelUsable,
  characterModelUsable,
  documentModelUsable,
  generationModelsForType,
  downloadOffersFor,
  hasUsableModelFor,
  imageModelUsable,
  poseModelUsable,
  supportedControlModes,
  videoModelUsable,
  visionCaptionModelUsable,
} from "./modelEligibility.js";
import { VISION_CAPTION_MODEL_ID, fallbackModels } from "./constants.js";

const caps = DEFAULT_MAC_CAPABILITIES; // gating off → Mac blocks are no-ops

describe("modelEligibility predicates", () => {
  it("imageModelUsable matches image models serving a mode, rejects other types", () => {
    expect(imageModelUsable({ type: "image", capabilities: ["text_to_image"] }, caps)).toBe(true);
    expect(imageModelUsable({ type: "image", capabilities: ["edit_image"] }, caps)).toBe(true);
    expect(imageModelUsable({ type: "image", capabilities: [] }, caps)).toBe(false);
    expect(imageModelUsable({ type: "video", capabilities: ["text_to_image"] }, caps)).toBe(false);
  });

  it("videoModelUsable matches video models with a video capability", () => {
    expect(videoModelUsable({ type: "video", capabilities: ["text_to_video"] }, caps)).toBe(true);
    expect(videoModelUsable({ type: "video", capabilities: ["animate_character"] }, caps)).toBe(true);
    expect(videoModelUsable({ type: "video", capabilities: [] }, caps)).toBe(false);
    expect(videoModelUsable({ type: "image", capabilities: ["text_to_video"] }, caps)).toBe(false);
  });

  it("documentModelUsable requires an interleave-capable image model", () => {
    expect(documentModelUsable({ type: "image", capabilities: ["interleave"] }, caps)).toBe(true);
    expect(documentModelUsable({ type: "image", capabilities: ["text_to_image"] }, caps)).toBe(false);
  });

  it("angle/pose predicates read the ui flags", () => {
    expect(angleModelUsable({ ui: { viewAngles: [{ id: "front" }] } }, caps)).toBe(true);
    expect(angleModelUsable({ ui: { viewAngles: [] } }, caps)).toBe(false);
    expect(poseModelUsable({ ui: { poseLibrary: true } }, caps)).toBe(true);
    expect(poseModelUsable({ ui: {} }, caps)).toBe(false);
    expect(characterModelUsable({ ui: { poseLibrary: true } }, caps)).toBe(true);
    expect(characterModelUsable({ ui: { viewAngles: [{ id: "front" }] } }, caps)).toBe(true);
    expect(characterModelUsable({ ui: {} }, caps)).toBe(false);
  });

  it("hasUsableModelFor counts complete models, not missing or torn ones", () => {
    const installed = { id: "b", type: "image", capabilities: ["text_to_image"], installState: "installed" };
    const incomplete = { id: "c", type: "image", capabilities: ["text_to_image"], installState: "incomplete" };
    const missing = { id: "a", type: "image", capabilities: ["text_to_image"], installState: "missing" };
    expect(hasUsableModelFor([missing, installed], imageModelUsable, caps)).toBe(true);
    expect(hasUsableModelFor([incomplete], imageModelUsable, caps)).toBe(false);
    expect(hasUsableModelFor([missing], imageModelUsable, caps)).toBe(false);
  });

  it("generation pickers retain usable stale models and exclude missing or torn models", () => {
    const models = [
      { id: "stale-image", type: "image", installState: "installed", updateAvailable: true },
      { id: "torn-image", type: "image", installState: "incomplete" },
      { id: "missing-image", type: "image", installState: "missing" },
      { id: "stale-video", type: "video", installState: "installed", updateAvailable: true },
    ];
    expect(generationModelsForType(models, "image").map((model) => model.id)).toEqual(["stale-image"]);
    expect(generationModelsForType(models, "video").map((model) => model.id)).toEqual(["stale-video"]);
  });

  it.each([
    ["installed current", "installed", false, true],
    ["installed usable-stale", "installed", true, true],
    ["missing", "missing", false, false],
    ["torn/incomplete", "incomplete", false, false],
  ])("screen gates classify %s consistently in every Studio mode", (_label, installState, updateAvailable, expected) => {
    const cases = [
      [{ id: "image", type: "image", capabilities: ["text_to_image"], installState, updateAvailable }, imageModelUsable],
      [{ id: "video", type: "video", capabilities: ["text_to_video"], installState, updateAvailable }, videoModelUsable],
      [{ id: "document", type: "image", capabilities: ["interleave"], installState, updateAvailable }, documentModelUsable],
      [{ id: "angle", ui: { viewAngles: [{ id: "front" }] }, installState, updateAvailable }, angleModelUsable],
      [{ id: "pose", ui: { poseLibrary: true }, installState, updateAvailable }, poseModelUsable],
      [{ id: "character", ui: { poseLibrary: true }, installState, updateAvailable }, characterModelUsable],
      [{ id: VISION_CAPTION_MODEL_ID, type: "utility", macOnly: false, installState, updateAvailable }, visionCaptionModelUsable],
    ];
    for (const [model, predicate] of cases) {
      expect(hasUsableModelFor([model], predicate, caps), `${model.id} gate`).toBe(expected);
    }
  });

  // SD3.5 surfacing + eligibility/gating (epic 7841 / sc-7873). The three native MLX variants are
  // text-to-image image models, so they are usable on Image Studio (text_to_image mode) when their
  // macSupport oracle reports supported. Under active Mac gating an unsupported variant (e.g. one
  // without an MLX engine, or any model off-Mac) is blocked from the picker; with gating off the Mac
  // blocks are no-ops so they always surface (Image Studio is the macOnly-aware path).
  it("imageModelUsable surfaces the SD3.5 variants and respects Mac gating", () => {
    const activeCaps = { ...DEFAULT_MAC_CAPABILITIES, macGatingActive: true, platform: "macos" };
    for (const id of ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"]) {
      const supported = {
        id,
        type: "image",
        capabilities: ["text_to_image", "style_variations"],
        macSupport: { supported: true, features: {} },
      };
      // Mac-supported native MLX variant → usable on Image Studio under active gating.
      expect(imageModelUsable(supported, activeCaps)).toBe(true);
      // Gating off (non-Mac / observe mode) → Mac block is a no-op, still usable.
      expect(imageModelUsable(supported, caps)).toBe(true);
      // Unsupported (no MLX engine for this variant) → hidden from the picker under active gating.
      const unsupported = { ...supported, macSupport: { supported: false } };
      expect(imageModelUsable(unsupported, activeCaps)).toBe(false);
    }
  });

  // Reference-image vision captioner gate (epic 8102, sc-8110; cross-platform via epic 8103, sc-8116).
  // The captioner is a single pinned utility model; usability = "this IS that model AND it can run
  // here". As of sc-8116 the catalog flips macOnly:false (the candle qwen3_vl vision tower landed in
  // candle-llm sc-8080), so the feature lights up on Windows/Linux too; the macOnly guard is kept
  // defensively for any future macOnly:true entry.
  it("visionCaptionModelUsable matches only the captioner model and is cross-platform (macOnly:false)", () => {
    const captioner = { id: VISION_CAPTION_MODEL_ID, type: "utility", macOnly: false };
    // Usable on every platform now (macOS / Windows / Linux) + pre-load empty platform.
    expect(visionCaptionModelUsable(captioner, { ...caps, platform: "macos" })).toBe(true);
    expect(visionCaptionModelUsable(captioner, { ...caps, platform: "windows" })).toBe(true);
    expect(visionCaptionModelUsable(captioner, { ...caps, platform: "linux" })).toBe(true);
    expect(visionCaptionModelUsable(captioner, caps)).toBe(true); // platform "" → no-op pre-load
    // Defensive macOnly guard: a macOnly:true entry still hides off Mac, surfaces on Mac.
    const macOnlyCaptioner = { ...captioner, macOnly: true };
    expect(visionCaptionModelUsable(macOnlyCaptioner, { ...caps, platform: "windows" })).toBe(false);
    expect(visionCaptionModelUsable(macOnlyCaptioner, { ...caps, platform: "macos" })).toBe(true);
    // A different model id is never the captioner.
    expect(visionCaptionModelUsable({ id: "some_other_model", macOnly: false }, { ...caps, platform: "macos" })).toBe(false);
    // Active Mac gating with the model's MLX oracle reporting unsupported → blocked.
    const blockedCaps = { ...DEFAULT_MAC_CAPABILITIES, macGatingActive: true, platform: "macos" };
    const unsupported = { ...captioner, macSupport: { supported: false } };
    expect(visionCaptionModelUsable(unsupported, blockedCaps)).toBe(false);
  });

  it("hasUsableModelFor / downloadOffersFor drive the captioner gate (sc-8110, cross-platform sc-8116)", () => {
    const macCaps = { ...caps, platform: "macos" };
    const installed = { id: VISION_CAPTION_MODEL_ID, type: "utility", macOnly: false, installState: "installed" };
    const missing = { id: VISION_CAPTION_MODEL_ID, type: "utility", macOnly: false, installState: "missing", recommended: true };
    // Present (installed) → screen is "ready".
    expect(hasUsableModelFor([installed], visionCaptionModelUsable, macCaps)).toBe(true);
    // Absent (missing) → not ready, and it surfaces as a recommended-first download offer.
    expect(hasUsableModelFor([missing], visionCaptionModelUsable, macCaps)).toBe(false);
    expect(downloadOffersFor([missing], visionCaptionModelUsable, macCaps).map((m) => m.id)).toEqual([
      VISION_CAPTION_MODEL_ID,
    ]);
    // On Windows the captioner is now usable too (epic 8103), so it surfaces the same download offer.
    expect(
      downloadOffersFor([missing], visionCaptionModelUsable, { ...caps, platform: "windows" }).map((m) => m.id),
    ).toEqual([VISION_CAPTION_MODEL_ID]);
  });

  it("supportedControlModes gates on ui.controlModes, canonical-ordered + deduped", () => {
    // A backbone advertising all three → all three, canonical order regardless of declared order.
    expect(supportedControlModes({ ui: { controlModes: ["depth", "pose", "canny"] } })).toEqual([
      "pose",
      "canny",
      "depth",
    ]);
    // Pose-only backbone → only pose (the picker would show a single tab).
    expect(supportedControlModes({ ui: { controlModes: ["pose"] } })).toEqual(["pose"]);
    // Canny+depth (no pose) → exactly those, in canonical order.
    expect(supportedControlModes({ ui: { controlModes: ["depth", "canny"] } })).toEqual(["canny", "depth"]);
    // Unknown modes are dropped (the worker only admits pose/canny/depth); dupes collapse.
    expect(supportedControlModes({ ui: { controlModes: ["pose", "POSE", "scribble", "canny"] } })).toEqual([
      "pose",
      "canny",
    ]);
    // No controlModes / no ui → empty (the panel hides).
    expect(supportedControlModes({ ui: {} })).toEqual([]);
    expect(supportedControlModes({})).toEqual([]);
    expect(supportedControlModes(null)).toEqual([]);
  });

  it("downloadOffersFor prefers recommended, falls back to any eligible, skips installed", () => {
    const models = [
      { id: "rec", type: "image", capabilities: ["text_to_image"], installState: "missing", recommended: true },
      { id: "plain", type: "image", capabilities: ["text_to_image"], installState: "missing" },
      { id: "done", type: "image", capabilities: ["text_to_image"], installState: "installed", recommended: true },
    ];
    expect(downloadOffersFor(models, imageModelUsable, caps).map((m) => m.id)).toEqual(["rec"]);
    // No recommended among eligible → fall back to all eligible (not installed).
    const noRec = models.filter((m) => m.id === "plain");
    expect(downloadOffersFor(noRec, imageModelUsable, caps).map((m) => m.id)).toEqual(["plain"]);
  });
});

// Audio Studio eligibility (epic 13400, sc-13403). audioModelServesMode is capability-driven: it
// reads only the model's `audio` sub-block (voices / editModes / conditioning / sampleRates), never
// the id. The five A2-seeded models must each map to EXACTLY one of speech/sfx/music/voiceclone and
// fail the other three, so the assertions below discriminate (a model must reject wrong modes).
describe("audio model eligibility (sc-13403)", () => {
  // Minimal fixtures mirroring the `audio` sub-blocks of the five seeded catalog models.
  const kokoro = {
    id: "kokoro_82m",
    type: "audio",
    audio: { voices: [{ id: "af_heart" }, { id: "am_michael" }], sampleRates: [24000], languages: ["en-US"], maxDurationSecs: 30 },
  };
  const moss = {
    id: "moss_sfx_v2",
    type: "audio",
    audio: { sampleRates: [48000], languages: ["en", "zh"], maxDurationSecs: 30 },
  };
  const acestep = {
    id: "acestep_v15_turbo",
    type: "audio",
    audio: { sampleRates: [48000], editModes: ["inpaint", "repaint", "extend"], conditioning: ["AudioEdit"], maxDurationSecs: 600 },
  };
  const openvoice = {
    id: "openvoice_v2",
    type: "audio",
    audio: { sampleRates: [22050], conditioning: ["ReferenceAudio"] },
  };
  const chatterbox = {
    id: "chatterbox_ve",
    type: "audio",
    audio: { conditioning: ["VoiceEmbedding"] },
  };
  // Native clone-TTS generator (sc-13412): ReferenceAudio + VoiceEmbedding, no voices/editModes → it
  // serves ONLY voiceclone (a text→waveform clone generator), exactly like the converter/embedder.
  const chatterboxTts = {
    id: "chatterbox_tts",
    type: "audio",
    audio: {
      languages: ["en", "en-US"],
      sampleRates: [24000],
      maxDurationSecs: 30,
      conditioning: ["VoiceEmbedding", "ReferenceAudio"],
    },
  };
  // Streaming TTS (sc-13675): NO voice bank — it serves "speech" via audio.supportsStreaming, and
  // must stay OFF "sfx" (the residual generator) despite advertising sampleRates.
  const mossTtsRealtime = {
    id: "moss_tts_realtime",
    type: "audio",
    audio: { languages: ["en", "zh"], sampleRates: [24000], maxDurationSecs: 2400, supportsStreaming: true },
  };
  // Multi-speaker dialogue TTS (sc-13676): NO voice bank — it serves "speech" via
  // audio.supportsMultiSpeaker (+ maxSpeakers), and must stay OFF "sfx" despite advertising sampleRates.
  const mossTtsd = {
    id: "moss_ttsd_v05",
    type: "audio",
    audio: { languages: ["zh", "en"], sampleRates: [24000], maxDurationSecs: 300, supportsMultiSpeaker: true, maxSpeakers: 2 },
  };

  const seeded = [
    ["Kokoro-82M", kokoro, "speech"],
    ["MOSS-TTS-Realtime (streaming)", mossTtsRealtime, "speech"],
    ["MOSS-TTSD (multi-speaker)", mossTtsd, "speech"],
    ["MOSS-SoundEffect-v2", moss, "sfx"],
    ["ACE-Step v1.5 Turbo", acestep, "music"],
    ["OpenVoice V2", openvoice, "voiceclone"],
    ["Chatterbox-VE", chatterbox, "voiceclone"],
    ["Chatterbox Clone-TTS", chatterboxTts, "voiceclone"],
  ];

  it("exposes the four Audio Studio mode keys in order", () => {
    expect(AUDIO_MODES).toEqual(["speech", "music", "sfx", "voiceclone"]);
  });

  it.each(seeded)("%s serves exactly its capability-derived mode and rejects the others", (_label, model, expectedMode) => {
    expect(audioModelServesMode(model, expectedMode), `${model.id} should serve ${expectedMode}`).toBe(true);
    for (const mode of AUDIO_MODES.filter((m) => m !== expectedMode)) {
      expect(audioModelServesMode(model, mode), `${model.id} must NOT serve ${mode}`).toBe(false);
    }
  });

  it("ACE-Step is music (editModes) and NOT voiceclone (its conditioning is AudioEdit, not a voice signal)", () => {
    expect(audioModelServesMode(acestep, "music")).toBe(true);
    expect(audioModelServesMode(acestep, "voiceclone")).toBe(false);
    // MOSS is the residual generator (sfx) — not music, because it advertises no editModes.
    expect(audioModelServesMode(moss, "music")).toBe(false);
    expect(audioModelServesMode(moss, "sfx")).toBe(true);
  });

  it("MOSS-TTS-Realtime serves speech via supportsStreaming (no voice bank) and NOT sfx", () => {
    // The streaming TTS has no voices — the streaming capability is its speech signal (sc-13675).
    expect(audioModelServesMode(mossTtsRealtime, "speech")).toBe(true);
    // It must NOT leak into the residual sfx bucket even though it advertises sampleRates.
    expect(audioModelServesMode(mossTtsRealtime, "sfx")).toBe(false);
    expect(audioModelServesMode(mossTtsRealtime, "music")).toBe(false);
    expect(audioModelServesMode(mossTtsRealtime, "voiceclone")).toBe(false);
    // And a plain (non-streaming) voiceless generator with the SAME sample-rate block stays sfx —
    // proving the classifier keys on the streaming flag, not on the absence of voices alone.
    const plainSfx = { id: "x", type: "audio", audio: { sampleRates: [24000], languages: ["en"] } };
    expect(audioModelServesMode(plainSfx, "speech")).toBe(false);
    expect(audioModelServesMode(plainSfx, "sfx")).toBe(true);
  });

  it("MOSS-TTSD serves speech via supportsMultiSpeaker (no voice bank) and NOT sfx", () => {
    // The multi-speaker dialogue TTS has no voices — the multi-speaker capability is its speech
    // signal (sc-13676), exactly as streaming is MOSS-TTS-Realtime's.
    expect(audioModelServesMode(mossTtsd, "speech")).toBe(true);
    // It must NOT leak into the residual sfx bucket even though it advertises sampleRates.
    expect(audioModelServesMode(mossTtsd, "sfx")).toBe(false);
    expect(audioModelServesMode(mossTtsd, "music")).toBe(false);
    expect(audioModelServesMode(mossTtsd, "voiceclone")).toBe(false);
    // A plain (non-multi-speaker) voiceless generator with the SAME sample-rate block stays sfx —
    // proving the classifier keys on the multi-speaker flag, not on the absence of voices alone.
    const plainSfx = { id: "y", type: "audio", audio: { sampleRates: [24000], languages: ["en"] } };
    expect(audioModelServesMode(plainSfx, "speech")).toBe(false);
    expect(audioModelServesMode(plainSfx, "sfx")).toBe(true);
  });

  it("audioModelServesMode is empty-block / unknown-mode safe", () => {
    expect(audioModelServesMode({ type: "audio" }, "speech")).toBe(false); // no audio block
    expect(audioModelServesMode({ type: "audio", audio: {} }, "sfx")).toBe(false); // empty block
    expect(audioModelServesMode(kokoro, "banana")).toBe(false); // unknown mode
    expect(audioModelServesMode(null, "speech")).toBe(false);
  });

  it("audioModelUsable matches audio models serving ≥1 mode, rejects other types + non-serving blocks", () => {
    for (const [, model] of seeded) {
      expect(audioModelUsable(model, caps), `${model.id} usable`).toBe(true);
    }
    // Wrong type → not usable even with an audio block.
    expect(audioModelUsable({ ...kokoro, type: "video" }, caps)).toBe(false);
    // Audio type but no serviceable capability (e.g. metadata-only block) → not usable.
    expect(audioModelUsable({ id: "bare", type: "audio", audio: { languages: ["en"] } }, caps)).toBe(false);
    expect(audioModelUsable({ id: "none", type: "audio" }, caps)).toBe(false);
  });

  it("audioModels resolve from a live catalog and from fallbackModels", () => {
    // Live-catalog fixture: installed audio entries surface; missing/torn/non-audio are excluded.
    const liveCatalog = [
      { ...kokoro, installState: "installed" },
      { ...acestep, installState: "installed", updateAvailable: true },
      { ...moss, installState: "missing" },
      { id: "some-video", type: "video", installState: "installed" },
    ];
    expect(generationModelsForType(liveCatalog, "audio").map((m) => m.id)).toEqual(["kokoro_82m", "acestep_v15_turbo"]);

    // Fallback mirror: the constants.js audio entries resolve the same models, and each still maps to
    // its correct capability-driven mode (proves the fallback carries the discriminating fields).
    const fallbackAudio = generationModelsForType(fallbackModels, "audio");
    expect(fallbackAudio.map((m) => m.id).sort()).toEqual(
      [
        "acestep_v15_turbo",
        "chatterbox_tts",
        "chatterbox_ve",
        "kokoro_82m",
        "moss_sfx_v2",
        "moss_tts_realtime",
        "moss_ttsd_v05",
        "openvoice_v2",
      ].sort(),
    );
    const expectedMode = {
      kokoro_82m: "speech",
      moss_tts_realtime: "speech",
      moss_ttsd_v05: "speech",
      moss_sfx_v2: "sfx",
      acestep_v15_turbo: "music",
      openvoice_v2: "voiceclone",
      chatterbox_ve: "voiceclone",
      chatterbox_tts: "voiceclone",
    };
    for (const model of fallbackAudio) {
      expect(audioModelServesMode(model, expectedMode[model.id]), `fallback ${model.id}`).toBe(true);
      for (const mode of AUDIO_MODES.filter((m) => m !== expectedMode[model.id])) {
        expect(audioModelServesMode(model, mode), `fallback ${model.id} must NOT serve ${mode}`).toBe(false);
      }
    }
    // Kokoro is the recommended default in the fallback list.
    expect(fallbackAudio.find((m) => m.id === "kokoro_82m")?.recommended).toBe(true);
  });
});
