import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import JSON5 from "json5";
import { describe, expect, it } from "vitest";
import { DEFAULT_MAC_CAPABILITIES } from "./macGating.js";
import {
  AUDIO_MODES,
  VIDEO_MODES,
  VECTOR_MODES,
  angleModelUsable,
  audioModelServesMode,
  audioModelUsable,
  characterModelUsable,
  documentModelUsable,
  generationModelsForType,
  downloadOffersFor,
  hasUsableModelFor,
  imageModelServesMode,
  imageModelUsable,
  missingRequiredModels,
  poseModelUsable,
  supportedControlModes,
  videoModelServesMode,
  videoModelUsable,
  visionCaptionModelUsable,
  vectorModelAvailability,
  vectorModelServesMode,
  vectorModelUsable,
} from "./modelEligibility.js";
import { VISION_CAPTION_MODEL_ID, fallbackModels } from "./constants.js";

const caps = DEFAULT_MAC_CAPABILITIES; // gating off → Mac blocks are no-ops

describe("modelEligibility predicates", () => {
  it("routes StarVector image_to_svg by backend and offers missing installs", () => {
    const model = {
      id: "starvector_1b",
      type: "vector",
      capabilities: ["image_to_svg"],
      vector: { providers: {
        mlx: { id: "mlx-starvector-1b", available: true },
        candle: { id: "candle-starvector-1b", available: true },
      } },
      installState: "installed",
      cacheState: "complete",
    };
    expect(VECTOR_MODES).toEqual(["image_to_svg", "text_to_svg"]);
    expect(vectorModelServesMode(model, "image_to_svg", { platform: "macos" })).toBe(true);
    expect(vectorModelServesMode(model, "text_to_svg", { platform: "macos" })).toBe(false);
    expect(vectorModelAvailability(model, "image_to_svg", { platform: "macos" })).toMatchObject({
      available: true,
      backend: "mlx",
      providerId: "mlx-starvector-1b",
    });
    const missing = { ...model, installState: "missing", cacheState: "missing", downloadable: true };
    expect(vectorModelAvailability(missing, "image_to_svg", { platform: "win32" })).toMatchObject({
      available: false,
      reason: "model_missing",
      backend: "candle",
    });
    expect(downloadOffersFor([missing], vectorModelUsable, { platform: "win32" })).toEqual([missing]);
    const unavailable = {
      ...model,
      vector: { providers: { ...model.vector.providers, candle: {
        id: "candle-starvector-1b",
        available: false,
        reason: "pending_terminal_inference_pin",
      } } },
    };
    expect(vectorModelAvailability(unavailable, "image_to_svg", { platform: "win32" })).toMatchObject({
      available: false,
      reason: "pending_terminal_inference_pin",
      backend: "candle",
    });
  });

  it("imageModelUsable matches image models serving a mode, rejects other types", () => {
    expect(imageModelUsable({ type: "image", capabilities: ["text_to_image"] }, caps)).toBe(true);
    expect(imageModelUsable({ type: "image", capabilities: ["edit_image"] }, caps)).toBe(true);
    expect(imageModelUsable({ type: "image", capabilities: [] }, caps)).toBe(false);
    expect(imageModelUsable({ type: "video", capabilities: ["text_to_image"] }, caps)).toBe(false);
    expect(imageModelUsable({ type: "image", capabilities: ["style_variations"] }, caps)).toBe(false);
  });

  // sc-13634 (#1780) made ImageStudio's picker treat an ABSENT capability array as a legacy
  // record that served Text; this shared mirror had kept the old `?? []` reading, so the two
  // disagreed for exactly those records. The distinction that matters is absent vs explicit:
  // an explicit [] (or an edit-only list) must still never serve Text.
  it("imageModelServesMode treats an ABSENT capability array as legacy Text, but not an explicit one", () => {
    expect(imageModelServesMode({ type: "image" }, "text_to_image", caps)).toBe(true);
    expect(imageModelUsable({ type: "image" }, caps)).toBe(true);

    expect(imageModelServesMode({ type: "image", capabilities: [] }, "text_to_image", caps)).toBe(false);
    expect(
      imageModelServesMode({ type: "image", capabilities: ["edit_image"] }, "text_to_image", caps),
    ).toBe(false);
    // A legacy record still only serves Text — the other three modes need a real declaration.
    expect(imageModelServesMode({ type: "image" }, "edit_image", caps)).toBe(false);
    expect(imageModelServesMode({ type: "image" }, "character_image", caps)).toBe(false);
    expect(imageModelServesMode({ type: "image" }, "style_variations", caps)).toBe(false);
  });

  it("videoModelUsable matches video models with a video capability", () => {
    expect(videoModelUsable({ type: "video", capabilities: ["text_to_video"] }, caps)).toBe(true);
    expect(videoModelUsable({ type: "video", capabilities: ["animate_character"] }, caps)).toBe(true);
    expect(videoModelUsable({ type: "video", capabilities: [] }, caps)).toBe(false);
    expect(videoModelUsable({ type: "image", capabilities: ["text_to_video"] }, caps)).toBe(false);
    const eros = { type: "video", macOnly: true, capabilities: ["text_to_video"] };
    expect(videoModelUsable(eros, { ...caps, platform: "macos" })).toBe(true);
    expect(videoModelUsable(eros, { ...caps, platform: "windows" })).toBe(false);
    expect(videoModelUsable(eros, { ...caps, platform: "linux" })).toBe(false);
    expect(videoModelUsable(eros, { ...caps, platform: "" })).toBe(false);
  });

  it("SC-18902 withdraws Eros from off-Mac pickers and offers without hiding base LTX", () => {
    const windows = { ...caps, platform: "windows" };
    const base = {
      id: "ltx_2_3",
      type: "video",
      capabilities: ["text_to_video"],
      installState: "installed",
      downloadable: true,
    };
    const eros = {
      id: "ltx_2_3_eros",
      type: "video",
      macOnly: true,
      capabilities: ["text_to_video"],
      installState: "missing",
      downloadable: false,
      usable: false,
      recommended: true,
    };
    expect(generationModelsForType([base, eros], "video").map((model) => model.id)).toEqual(["ltx_2_3"]);
    expect(downloadOffersFor([eros], videoModelUsable, windows)).toEqual([]);
    expect(videoModelUsable(base, windows)).toBe(true);

    const unknownFallback = fallbackModels.filter((model) =>
      videoModelUsable(model, { ...caps, platform: "" }),
    );
    expect(unknownFallback.some((model) => model.id === "ltx_2_3_eros")).toBe(false);
    expect(unknownFallback.some((model) => model.id === "ltx_2_3")).toBe(true);

    const macEros = { ...eros, installState: "missing", downloadable: true, usable: true };
    expect(downloadOffersFor([macEros], videoModelUsable, { ...caps, platform: "macos" }).map((model) => model.id)).toEqual([
      "ltx_2_3_eros",
    ]);
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

  it("re-lights an imported pose picker when any active MLX provider route serves pose", () => {
    const activeMac = { ...DEFAULT_MAC_CAPABILITIES, macGatingActive: true, platform: "macos" };
    const importedKrea = {
      id: "user_kreamania",
      type: "image",
      family: "krea_2",
      ui: { poseLibrary: true, poseControlScale: true, controlModes: ["pose"] },
      macSupport: { supported: true, features: { pose: true } },
    };
    expect(poseModelUsable(importedKrea, activeMac)).toBe(true);

    expect(
      poseModelUsable(
        { ...importedKrea, macSupport: { supported: true, features: { pose: false } } },
        activeMac,
      ),
    ).toBe(false);
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
        capabilities: ["text_to_image"],
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

  it("downloadOffersFor prefers recommended and skips installed or unavailable entries", () => {
    const models = [
      { id: "rec", type: "image", capabilities: ["text_to_image"], installState: "missing", recommended: true },
      { id: "plain", type: "image", capabilities: ["text_to_image"], installState: "missing" },
      { id: "done", type: "image", capabilities: ["text_to_image"], installState: "installed", recommended: true },
      { id: "unusable", type: "image", capabilities: ["text_to_image"], installState: "missing", usable: false, recommended: true },
      { id: "unavailable", type: "image", capabilities: ["text_to_image"], installState: "missing", downloadable: false, recommended: true },
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

describe("missingRequiredModels", () => {
  const catalog = [
    { id: "person_detector", name: "YOLO11m Person Detector", installState: "installed" },
    { id: "dwpose_pose_detector", name: "DWPose Pose Detector", installState: "missing" },
    { id: "torn_one", name: "Torn", installState: "incomplete" },
  ];

  it("returns only the entries that are not install-complete", () => {
    expect(missingRequiredModels(catalog, ["person_detector", "dwpose_pose_detector"])).toEqual([
      catalog[1],
    ]);
  });

  it("treats a torn install as missing — it fails to load exactly the same way", () => {
    expect(missingRequiredModels(catalog, ["torn_one"]).map((m) => m.id)).toEqual(["torn_one"]);
  });

  it("preserves the caller's declaration order", () => {
    const ids = ["torn_one", "dwpose_pose_detector"];
    expect(missingRequiredModels(catalog, ids).map((m) => m.id)).toEqual(ids);
  });

  // The load-bearing rule, shared with the Pose Library gate: `models` is [] on first render and an
  // older API may not declare a utility entry at all, while the worker still resolves it. Reading
  // "absent" as "missing" would block a working install on every mount.
  it("treats an id with no catalog entry as satisfied", () => {
    expect(missingRequiredModels(catalog, ["not_in_catalog"])).toEqual([]);
    expect(missingRequiredModels([], ["dwpose_pose_detector"])).toEqual([]);
  });

  it("tolerates missing arguments", () => {
    expect(missingRequiredModels(undefined, undefined)).toEqual([]);
    expect(missingRequiredModels(catalog, undefined)).toEqual([]);
  });
});

// THE CLASS GUARD, web half (sc-17159, GH #2074). `VIDEO_MODES` is a reachability gate, not a
// display list: `videoModelUsable` requires a model to serve at least one mode IN THIS ARRAY, and
// `VideoStudio.jsx` falls a recipe back to `text_to_video` for any mode it does not contain. So a
// capability the shipped manifest advertises but this array omits makes the model unusable in the
// Video Studio however completely the server is wired.
//
// Read off BOTH real sources, like its Rust siblings — the advertisement from the shipped
// `builtin.models.jsonc` bytes, the admission from the real exported `VIDEO_MODES`:
//   * `every_declared_video_capability_is_submittable` (apps/rust-api) — the API allow-list;
//   * `every_declared_video_capability_is_claimable_by_some_lane` (sceneworks-core) — the lanes.
// The `fallbackModels` mirror is checked too, because App.jsx serves it to the real picker before
// the catalog loads, and a mirror that drops a capability hides a mode for that whole window.
describe("declared video capabilities are all offerable", () => {
  const HERE = dirname(fileURLToPath(import.meta.url));
  const manifestPath = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");
  const manifest = JSON5.parse(readFileSync(manifestPath, "utf8"));
  const shipped = (Array.isArray(manifest) ? manifest : manifest.models).filter(
    (entry) => entry.type === "video",
  );

  it("reads a real video catalog, so the assertions below are not vacuous", () => {
    expect(shipped.length).toBeGreaterThanOrEqual(12);
    expect(shipped.flatMap((entry) => entry.capabilities ?? []).length).toBeGreaterThanOrEqual(30);
  });

  it("VIDEO_MODES admits every capability the shipped manifest advertises", () => {
    for (const entry of shipped) {
      expect(Array.isArray(entry.capabilities), `${entry.id} declares capabilities`).toBe(true);
      for (const mode of entry.capabilities) {
        expect(
          VIDEO_MODES,
          `${entry.id} advertises "${mode}" but VIDEO_MODES omits it, so videoModelServesMode can ` +
            `never return true for it and the model is dropped from the Video Studio entirely`,
        ).toContain(mode);
      }
    }
  });

  it("the fallbackModels mirror declares the same capabilities as the manifest", () => {
    for (const entry of shipped) {
      const mirrored = fallbackModels.find((model) => model.id === entry.id);
      if (!mirrored) continue; // the mirror is deliberately partial; only drift is the defect.
      expect(
        [...(mirrored.capabilities ?? [])].sort(),
        `${entry.id}: the constants.js mirror the picker uses before the catalog loads must not ` +
          `drop or invent a capability`,
      ).toEqual([...entry.capabilities].sort());
    }
  });

  // sc-19504. The withdrawal, asserted where it is actually LOAD-BEARING.
  //
  // On a Mac the First/Last tab was already hidden for this model — `macSupport.features.videoModes`
  // is built by mapping the real `video_mode_is_mlx_eligible`, which says false. OFF-Mac
  // `macGatingActive` is false, `macVideoModeBlock` is inert, and `videoModelServesMode` collapses
  // to `capabilities.includes(mode)` alone. So `capabilities` was the ONLY thing between a
  // Windows/Linux user and a tab whose every submission queued forever — which is why the fix is
  // the manifest array and not a gating flag.
  it("the 14B I2V no longer offers First/Last Frame — on any platform", () => {
    const entry = shipped.find((model) => model.id === "wan_2_2_i2v_14b");
    expect(entry, "wan_2_2_i2v_14b is a shipped video model").toBeTruthy();
    expect(entry.capabilities).not.toContain("first_last_frame");
    expect(entry.ui?.recommendedFor ?? []).not.toContain("first_last_frame");

    // Off-Mac: gating inactive, so this is `capabilities` speaking for itself.
    const offMac = { ...DEFAULT_MAC_CAPABILITIES, macGatingActive: false, platform: "win32" };
    expect(videoModelServesMode(entry, "first_last_frame", offMac)).toBe(false);
    // …and the modes a lane really does serve are untouched, so the withdrawal narrowed one
    // capability rather than the model. Without this the assertion above would pass on an entry
    // that had lost every capability.
    for (const mode of ["image_to_video", "extend_clip", "video_bridge"]) {
      expect(videoModelServesMode(entry, mode, offMac), `still serves ${mode}`).toBe(true);
    }
    expect(videoModelUsable({ ...entry, macSupport: undefined }, offMac)).toBe(true);
  });

  // sc-19570. THE OFF-MAC GATE, asserted with a FOREIGN-platform fixture.
  //
  // This is the same shape as the sc-19504 guard above, generalised: `capabilities` was the only
  // thing between a Windows/Linux user and a tab whose every submission queued forever, for
  // THIRTEEN advertised pairs rather than one. sc-19504 fixed its pair by withdrawing the
  // capability, which is not available here — every pair below renders correctly on a Mac, so the
  // manifest is right and the missing piece was a per-platform gate.
  //
  // The fixture is what makes this a real check. On the host these tests run on there is no
  // off-Mac anything: `candleGatingActive` arrives from `GET /api/v1/capabilities/mac` and is
  // false on a Mac, so every helper in candleGating.js is inert. Tagging the caps object with the
  // foreign platform is what runs the branch.
  it("an MLX-only video mode is not offered off-Mac, and is still offered on a Mac", () => {
    // The per-model block the API now emits, in the shape `model_candle_support` serializes.
    const candleSupport = (served) => ({
      supported: served.length > 0,
      features: { videoModes: Object.fromEntries(VIDEO_MODES.map((m) => [m, served.includes(m)])) },
    });
    const offMac = { ...DEFAULT_MAC_CAPABILITIES, candleGatingActive: true, platform: "linux" };
    const onMac = { ...DEFAULT_MAC_CAPABILITIES, macGatingActive: true, platform: "darwin" };

    // LTX-2.3: candle serves `text_to_video` and nothing else, and the Mac serves all six.
    const ltx = shipped.find((model) => model.id === "ltx_2_3");
    expect(ltx, "ltx_2_3 is a shipped video model").toBeTruthy();
    const ltxOffMac = { ...ltx, candleSupport: candleSupport(["text_to_video"]) };
    const ltxOnMac = {
      ...ltx,
      macSupport: { supported: true, features: { videoModes: Object.fromEntries(VIDEO_MODES.map((m) => [m, ltx.capabilities.includes(m)])) } },
      candleSupport: candleSupport(["text_to_video"]),
    };
    for (const mode of [
      "image_to_video",
      "first_last_frame",
      "extend_clip",
      "video_bridge",
      "replace_person",
    ]) {
      expect(
        videoModelServesMode(ltxOffMac, mode, offMac),
        `ltx_2_3 + ${mode} has no off-Mac lane and must not be offered there`,
      ).toBe(false);
      // …and is STILL offered on a Mac. Breaking the platform where it works, to fix the one where
      // it does not, is the failure this leg exists to catch.
      expect(
        videoModelServesMode(ltxOnMac, mode, onMac),
        `ltx_2_3 + ${mode} renders on a Mac and must stay offered there`,
      ).toBe(true);
    }
    // The mode candle DOES serve is untouched, so the gate narrowed the tab list rather than the
    // model — without this the assertions above would pass on a block that hid everything.
    expect(videoModelServesMode(ltxOffMac, "text_to_video", offMac)).toBe(true);
    expect(videoModelUsable(ltxOffMac, offMac)).toBe(true);

    // A model with NO off-Mac lane at all disappears from the picker rather than showing every tab
    // disabled: `wan_2_2_vace_fun_14b` advertises `replace_person` alone and candle does not route
    // it, so `candleSupport.supported` is false.
    const vaceFun = shipped.find((model) => model.id === "wan_2_2_vace_fun_14b");
    const vaceOffMac = { ...vaceFun, candleSupport: candleSupport([]) };
    expect(videoModelServesMode(vaceOffMac, "replace_person", offMac)).toBe(false);
    expect(videoModelUsable(vaceOffMac, offMac)).toBe(false);
    expect(downloadOffersFor([{ ...vaceOffMac, installState: "missing" }], videoModelUsable, offMac))
      .toEqual([]);

    // …and on a Mac, that same model is offered. Two platforms, two answers, one model.
    const vaceOnMac = {
      ...vaceFun,
      macSupport: { supported: true, features: { videoModes: { replace_person: true } } },
      candleSupport: candleSupport([]),
    };
    expect(videoModelServesMode(vaceOnMac, "replace_person", onMac)).toBe(true);
    expect(videoModelUsable(vaceOnMac, onMac)).toBe(true);
  });

  // The switch itself: with `candleGatingActive` false — a Mac, or any client before the
  // capabilities endpoint has answered — the block is inert even when it says `false`. A gate that
  // read `candleSupport` unconditionally would hide Mac-served tabs on the Mac.
  it("the candle block is inert until off-Mac gating is active", () => {
    const model = {
      type: "video",
      capabilities: ["image_to_video"],
      candleSupport: { supported: false, features: { videoModes: { image_to_video: false } } },
    };
    for (const inert of [
      DEFAULT_MAC_CAPABILITIES,
      { ...DEFAULT_MAC_CAPABILITIES, macGatingActive: true, platform: "darwin" },
    ]) {
      expect(videoModelServesMode(model, "image_to_video", inert)).toBe(true);
      expect(videoModelUsable(model, inert)).toBe(true);
    }
    const offMac = { ...DEFAULT_MAC_CAPABILITIES, candleGatingActive: true, platform: "win32" };
    expect(videoModelServesMode(model, "image_to_video", offMac)).toBe(false);
    expect(videoModelUsable(model, offMac)).toBe(false);
  });

  it("both MiniMax-H3 partitions are usable in the Video Studio, each on its own modes", () => {
    // The regression this pins: the family installs on macOS ONLY, so if the server ever answers
    // `macSupport.supported: false` for it again (it did until sc-17159 added the VIDEO_MODEL_CAPS
    // rows), `macModelBlock` drops it from the picker on the only platform it runs on.
    const gating = { ...DEFAULT_MAC_CAPABILITIES, macGatingActive: true, platform: "darwin" };
    for (const [id, served] of [
      ["minimax_h3", ["text_to_video", "image_to_video", "first_last_frame"]],
      ["minimax_h3_ref", ["reference_to_video"]],
    ]) {
      const entry = shipped.find((model) => model.id === id);
      const model = {
        ...entry,
        macSupport: {
          supported: true,
          features: {
            videoModes: Object.fromEntries(VIDEO_MODES.map((m) => [m, served.includes(m)])),
          },
        },
      };
      expect(videoModelUsable(model, gating), `${id} must be usable on a Mac`).toBe(true);
      for (const mode of VIDEO_MODES) {
        expect(videoModelServesMode(model, mode, gating), `${id} serves ${mode}?`).toBe(
          served.includes(mode),
        );
      }
      // …and blocked outright the moment the server says the Mac lane does not serve it, which is
      // what makes the assertion above about routing rather than about the manifest alone.
      expect(
        videoModelUsable({ ...model, macSupport: { supported: false, reason: null } }, gating),
      ).toBe(false);
    }
  });
});
