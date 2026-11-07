// Shared per-screen model-eligibility predicates (sc-5946). Each Studio already
// derived "can this model serve this screen?" inline; those predicates are lifted here
// verbatim so the per-Studio availability gates and the screens themselves agree on
// exactly which models qualify. Predicates are capability + Mac-gating only (no install
// state) — callers layer installState via the helpers at the bottom.
import { POSE_DETECT_MODEL_ID, VISION_CAPTION_MODEL_ID } from "./constants.js";

// The native fast recipe is packaged only with the two shipped Wan A14B engines. Imported
// same-family checkpoints do not inherit the worker's builtin Lightning route.
export function wanLightningAvailable(model) {
  return model?.family === "wan-video" && model?.type === "video" && !model?.importedSource &&
    model?.downloads?.some((download) => download.repo === "lightx2v/Wan2.2-Lightning") === true;
}

// Paired high/low-noise LoRA uploads belong to the two shipped 14B generation backbones.
// VACE Fun also has sizeClass 14b, but only advertises replace_person.
export function wanPairedLoraAvailable(model) {
  return model?.family === "wan-video" && model?.type === "video" && !model?.importedSource &&
    model?.loraCompatibility?.sizeClass === "14b" &&
    model?.capabilities?.some((mode) => mode === "text_to_video" || mode === "image_to_video") === true;
}

export function ltxIcLoraRequired(model, mode) {
  return model?.family === "ltx-video" && model?.type === "video" &&
    ["2.3", "2.5"].includes(model?.loraCompatibility?.generation) &&
    !model?.importedSource &&
    ["extend_clip", "video_bridge", "replace_person"].includes(mode);
}

// Candle's native video tier picker needs a shipped Windows/Linux variant ladder. LTX 2.3
// still uses its separate legacy quantization control; LTX 2.5 has the native tier ladder.
export function candleVideoTierPickerAvailable(model) {
  return model?.type === "video" && !model?.importedSource &&
    !(model?.family === "ltx-video" && model?.loraCompatibility?.generation === "2.3") &&
    model?.downloads?.some((download) => download.variant &&
      download.platforms?.some((platform) => platform === "windows" || platform === "linux")) === true;
}

// The shipped low-step SDXL checkpoint cannot run Hires.fix's second diffusion pass.
export function lowStepSdxlCheckpoint(model) {
  return model?.family === "sdxl" && model?.type === "image" &&
    model?.defaults?.steps === 5 && !Array.isArray(model?.limits?.samplers);
}

// Platform support is supplied by the catalog projection of typed routes and feature facts.
// Readiness switches keep all gates inert until capabilities arrive.
export const MAC_NOT_AVAILABLE_LABEL = "Not available on Mac (MLX only)";

// The inert default used until the capabilities endpoint responds (and the permanent value on
// non-Mac / observe-mode deployments). Every helper below short-circuits to "not blocked".
export const DEFAULT_MAC_CAPABILITIES = {
  macGatingActive: false,
  // The off-Mac switch is inert here for the same reason
  // `macGatingActive` is: until the capabilities endpoint responds, nothing is gated.
  candleGatingActive: false,
  platform: "",
  notAvailableLabel: MAC_NOT_AVAILABLE_LABEL,
  features: {},
  training: { supportedKernels: [], lokrOnWanSupported: false },
};

function label(caps) {
  return caps?.notAvailableLabel || MAC_NOT_AVAILABLE_LABEL;
}

// Compose the user-facing affordance text from a backend UnsupportedReason
// ({ feature, detail, suggestedEpic }).
export function macReasonText(caps, reason) {
  const base = label(caps);
  if (!reason) return base;
  const detail = reason.detail || reason.feature || "";
  const epic = reason.suggestedEpic ? ` (${reason.suggestedEpic})` : "";
  return detail ? `${base} — ${detail}${epic}` : base;
}

function featureLabel(feature) {
  switch (feature) {
    case "pose":
      return "pose conditioning";
    case "reference":
      return "reference / identity conditioning";
    case "edit":
      return "image editing";
    case "lycoris":
      return "third-party LyCORIS LoRA";
    default:
      return feature;
  }
}

// Whether gating is engaged at all (a Mac in MLX-required mode).
export function macGatingActive(caps) {
  return Boolean(caps?.macGatingActive);
}

// A whole model that can't run in the Rust/MLX flow on Mac (torch-only) → hidden/disabled in
// pickers. Returns `{ blocked, reason, text }` or `null`.
export function macModelBlock(model, caps) {
  if (!macGatingActive(caps)) return null;
  if (model?.macSupport?.supported === false) {
    const reason = model.macSupport.reason ?? null;
    return { blocked: true, reason, text: macReasonText(caps, reason) };
  }
  return null;
}

// A per-model feature control (pose / reference / edit / lycoris) the model can't run on MLX.
// Returns `{ blocked, text }` or `null`.
export function macModelFeatureBlock(model, caps, feature) {
  if (!macGatingActive(caps)) return null;
  const features = model?.macSupport?.features;
  if (features && features[feature] === false) {
    return {
      blocked: true,
      text: `${label(caps)} — this model's ${featureLabel(feature)} is not available in the native flow on Mac.`,
    };
  }
  return null;
}

// Whether a video_generate mode routes to MLX for the given (supported) video model.
export function macVideoModeBlock(model, caps, mode) {
  if (!macGatingActive(caps)) return null;
  const modes = model?.macSupport?.features?.videoModes;
  if (modes && modes[mode] === false) {
    return {
      blocked: true,
      // On Mac the runtime is MLX-only (epic 3482) — there is no torch fallback, so a
      // blocked mode simply isn't served for this model here (some modes have no MLX path
      // on this engine; other specialist surfaces may be exposed under a different catalog
      // model). Don't call it "torch-only".
      text: `${label(caps)} — the selected model doesn't support this mode on macOS.`,
    };
  }
  return null;
}

// A non-model Python surface (imageUpscale, poseFromPhoto, personDetect, datasetCaptioning).
// Returns `{ blocked, reason, text }` or `null`. Video modes are gated per-model via
// macVideoModeBlock, not here (sc-3773).
export function macFeatureBlock(caps, key) {
  if (!macGatingActive(caps)) return null;
  const feature = caps?.features?.[key];
  if (feature && feature.supported === false) {
    const reason = feature.reason ?? null;
    return { blocked: true, reason, text: macReasonText(caps, reason) };
  }
  return null;
}

// Whether a specific image_upscale engine should be hidden from the picker on this platform.
// Two engines are platform-restricted (ids match the worker):
//   * `aura-sr` — a torch-only GigaGAN DROPPED on Mac (sc-3668) and now dropped as an offered engine
//     on EVERY platform (sc-5499): it has no native (MLX/candle) path and the Python torch backend
//     that served it off-Mac is retired in Phase 7 (epic 5483), so the picker hides it everywhere
//     (users don't build on a path about to disappear). Gated off the platform-intrinsic
//     `imageUpscaleAuraSr` capability (`supported: false` on every platform), so — like SeedVR2 — the
//     check is gating-independent (hidden even pre-load, when caps haven't arrived). Real-ESRGAN is
//     the cross-platform default upscaler.
//   * `seedvr2` — the one-step diffusion upscaler (epic 4811 / sc-4815), the INVERSE of AuraSR:
//     backed by native MLX on Mac and the Candle CUDA/NVIDIA port on Windows (sc-5928) + Linux
//     (sc-5160 — candle is CPU+CUDA cross-platform, so Linux rides the Windows port). Gated off the
//     platform-intrinsic `imageUpscaleSeedvr2` capability (true on macOS + Windows + Linux, in any
//     mode), so it is hidden only pre-load (no caps yet) and shown on every GPU platform —
//     independent of the gating switch.
export function macUpscaleEngineBlocked(caps, engine) {
  if (engine === "seedvr2") {
    return caps?.features?.imageUpscaleSeedvr2?.supported !== true;
  }
  if (engine === "aura-sr") {
    return caps?.features?.imageUpscaleAuraSr?.supported !== true;
  }
  // Iris restoration (sc-25683): platform-intrinsic like SeedVR2 (native MLX on Mac, Candle on
  // Windows/Linux). Whether the task is INSTALLED is a separate gate (upscaleEngines.js).
  if (engine === "iris") {
    return caps?.features?.imageUpscaleIris?.supported !== true;
  }
  return false;
}

// Whether a training kernel has no native Rust trainer (so its base model is gated in Training).
export function macTrainingKernelBlocked(caps, kernel) {
  if (!macGatingActive(caps) || !kernel) return false;
  const supported = caps?.training?.supportedKernels ?? [];
  return !supported.includes(kernel);
}

// Partition a model list into the Mac-available ones (for the picker) — a no-op when gating is off.
export function macAvailableModels(models, caps) {
  if (!macGatingActive(caps)) return models ?? [];
  return (models ?? []).filter((model) => model?.macSupport?.supported !== false);
}

// The models hidden from a picker on Mac, so a screen can show a "N unavailable" affordance.
export function macBlockedModels(models, caps) {
  if (!macGatingActive(caps)) return [];
  return (models ?? []).filter((model) => model?.macSupport?.supported === false);
}

export const CANDLE_NOT_AVAILABLE_LABEL = "Not available on this platform (macOS/MLX only)";

// Whether off-Mac gating is engaged at all (the API host is Windows/Linux, so candle is the only
// lane that can register). Inert until the capabilities endpoint responds, mirroring
// `macGatingActive`.
export function candleGatingActive(caps) {
  return Boolean(caps?.candleGatingActive);
}

// A whole video model with no candle lane at all (no VIDEO_UI_MODES entry is candle-claimable) →
// hidden/disabled in pickers off-Mac. Returns `{ blocked, reason, text }` or `null`, the same shape
// `macModelBlock` returns.
export function candleModelBlock(model, caps) {
  if (!candleGatingActive(caps)) return null;
  if (model?.candleSupport?.supported === false) {
    const reason = model.candleSupport.reason ?? null;
    const detail = reason?.detail || reason?.feature || "";
    return {
      blocked: true,
      reason,
      text: detail
        ? `${CANDLE_NOT_AVAILABLE_LABEL} — ${detail}`
        : CANDLE_NOT_AVAILABLE_LABEL,
    };
  }
  return null;
}

// Whether a `video_generate` mode has a candle lane for the given video model. The off-Mac twin of
// `macVideoModeBlock`. Returns `{ blocked, text }` or `null`.
export function candleVideoModeBlock(model, caps, mode) {
  if (!candleGatingActive(caps)) return null;
  const modes = model?.candleSupport?.features?.videoModes;
  if (modes && modes[mode] === false) {
    return {
      blocked: true,
      // Off-Mac the runtime is candle-only — there is no MLX fallback and no torch fallback
      // (the Python backend was retired in Phase 7), so a blocked mode simply isn't served for
      // this model here. Name macOS, because that IS where the pair works: a user who sees this
      // is being told the combination exists, just not on this machine.
      text: `${CANDLE_NOT_AVAILABLE_LABEL} — this model only supports this mode on macOS.`,
    };
  }
  return null;
}

// Partition a model list into the ones with a candle lane (for the picker) — a no-op on Mac.
// A training target whose kernel no candle worker can run (sc-24161 — an MLX-only kernel) →
// disabled off-Mac, the twin of
// `macTrainingKernelBlocked`. Keyed on the server's `training.candleSupportedKernels` (the
// candle-routed set), never a hardcoded kernel list; inert when the field is absent.
export function candleTrainingKernelBlocked(caps, kernel) {
  if (!candleGatingActive(caps) || !kernel) return false;
  const supported = caps?.training?.candleSupportedKernels;
  return Array.isArray(supported) && !supported.includes(kernel);
}

export function candleAvailableModels(models, caps) {
  if (!candleGatingActive(caps)) return models ?? [];
  return (models ?? []).filter((model) => model?.candleSupport?.supported !== false);
}

// Image generation modes a model can serve (ImageStudio mode tabs).
const IMAGE_MODES = ["text_to_image", "edit_image", "character_image"];

// Video generation modes a model can advertise (VideoStudio modeOptions).
export const VIDEO_MODES = [
  "text_to_video",
  "image_to_video",
  "first_last_frame",
  "extend_clip",
  "video_bridge",
  "replace_person",
  "video_to_video",
  "reference_to_video",
  "reference_video_to_video",
  "multi_video_to_video",
  "ads2v",
  "animate_character",
];

// Audio generation modes the Audio Studio exposes (epic 13400 C0/C1 mirror these keys).
export const AUDIO_MODES = ["speech", "music", "sfx", "voiceclone"];

// Vector Studio modes. A manifest must declare the mode and the selected native backend provider;
// install state is evaluated separately so a missing-but-eligible model can be offered for download.
export const VECTOR_MODES = ["image_to_svg", "text_to_svg"];

export function vectorModelServesMode(model, mode, caps) {
  if (model?.type !== "vector" || !VECTOR_MODES.includes(mode) || !model?.capabilities?.includes(mode)) {
    return false;
  }
  const backend = caps?.platform === "macos" || caps?.platform === "darwin" ? "mlx" : "candle";
  return model?.vector?.providers?.[backend]?.available === true;
}

export function vectorModelAvailability(model, mode, caps) {
  if (model?.type !== "vector" || !VECTOR_MODES.includes(mode) || !model?.capabilities?.includes(mode)) {
    return { available: false, reason: "unsupported_mode" };
  }
  const backend = caps?.platform === "macos" || caps?.platform === "darwin" ? "mlx" : "candle";
  const provider = model?.vector?.providers?.[backend];
  if (provider?.available !== true) {
    return { available: false, reason: provider?.reason ?? "provider_not_linked", backend };
  }
  if (model?.installState !== "installed" || model?.cacheState !== "complete") {
    return {
      available: false,
      reason: model?.cacheState === "incomplete" ? "model_incomplete" : "model_missing",
      backend,
    };
  }
  return { available: true, reason: null, backend, providerId: provider.id };
}

export function vectorModelUsable(model, caps) {
  return model?.type === "vector" && VECTOR_MODES.some((mode) => vectorModelServesMode(model, mode, caps));
}

// Conditioning kinds that mark a voice-clone (voice-conversion / speaker-embedding) model —
// distinct from ACE-Step's "AudioEdit" conditioning, which is a music-editing signal, not a
// reference/identity signal. Compared case-insensitively so manifest casing never matters.
const VOICE_CLONE_CONDITIONING = new Set(["referenceaudio", "voiceembedding"]);

// Image Studio — mirrors ImageStudio.jsx `imageModelServesMode`. A model serves a mode
// when it declares the capability and, under active Mac gating, that feature is MLX-routed
// for it (the macModelFeatureBlock calls are no-ops off Mac).
export function imageModelServesMode(model, mode, caps) {
  // sc-13634 (#1780) taught ImageStudio's local copy that an ABSENT capability array is a
  // legacy installed-model record which historically meant "Text". This mirror kept the old
  // `?? []` reading, so the two authorities disagreed for exactly those records: the studio
  // offered them in Text while anything reading this predicate (the screen gate, download
  // offers, and the Simple Image Studio) treated them as serving nothing. An EXPLICIT list —
  // including `[]` — stays authoritative in both, so an edit-only or reference-only model
  // still can never leak into Text.
  const capabilitiesDeclared = Array.isArray(model?.capabilities);
  const capabilities = capabilitiesDeclared ? model.capabilities : [];
  if (!IMAGE_MODES.includes(mode)) return false;
  if (mode === "edit_image") {
    return (
      (capabilities.includes("edit_image") || capabilities.includes("image_edit")) &&
      !macModelFeatureBlock(model, caps, "edit")
    );
  }
  if (mode === "character_image") {
    return capabilities.includes("character_image") && !macModelFeatureBlock(model, caps, "reference");
  }
  return !capabilitiesDeclared || capabilities.includes("text_to_image");
}

// Usable on Image Studio: an image-type model that isn't Mac-blocked and serves ≥1 mode.
export function imageModelUsable(model, caps) {
  return (
    model?.type === "image" &&
    !macModelBlock(model, caps) &&
    IMAGE_MODES.some((mode) => imageModelServesMode(model, mode, caps))
  );
}

// Video Studio — the ONE authority for "does this model serve this mode?" (VideoStudio.jsx imports
// it rather than keeping the local copy it used to have; two copies is how one platform's gate got
// added and the other's did not).
//
// Three conjuncts, one per layer of the reachability question (sc-19570):
//   * the model DECLARES the mode (the manifest capability), and
//   * under active Mac gating, MLX claims it for this model (`macVideoModeBlock`), and
//   * under active off-Mac gating, candle claims it for this model (`candleVideoModeBlock`).
// The two platform blocks are mutually exclusive and each is a no-op on the other's platform, so AT
// MOST one of them can ever fire.
//
// ⚠️ NOT "exactly one", and declaration alone is NOT "never again the whole answer" — this comment
// asserted both and each is false in the same two states, where NEITHER block fires and this
// function collapses back to `capabilities.includes(mode)`:
//   1. Before `GET /api/v1/capabilities/mac` responds. `DEFAULT_MAC_CAPABILITIES` sets both
//      `macGatingActive` and `candleGatingActive` false, so both helpers short-circuit to "not
//      blocked" on every platform. Every consumer of this predicate renders at least one pass in
//      that window.
//   2. On a Mac in observe mode, permanently. `macGatingActive` is the SCENEWORKS_MLX_REQUIRED
//      rollout flag rather than a fact about the host, and `candleGatingActive` is `!is_mac`, so a
//      Mac with the flag off has neither.
// Both are deliberate — a client that has not been told what it is running on must not invent a
// gate — but a reader who believes the stronger claim will look for a bug that is not there.
export function videoModelServesMode(model, mode, caps) {
  return (
    Boolean(model?.capabilities?.includes(mode)) &&
    !macVideoModeBlock(model, caps, mode) &&
    !candleVideoModeBlock(model, caps, mode)
  );
}

// Usable on Video Studio: a video-type model that isn't platform-blocked and serves ≥1 mode.
export function videoModelUsable(model, caps) {
  const platform = caps?.platform ?? "";
  return (
    model?.type === "video" &&
    !macModelBlock(model, caps) &&
    !candleModelBlock(model, caps) &&
    !(model?.macOnly === true && platform !== "macos") &&
    VIDEO_MODES.some((mode) => videoModelServesMode(model, mode, caps))
  );
}

// Audio Studio — capability-driven per-mode eligibility, derived from the model's `audio`
// sub-block (mirrors how videoModelServesMode reads video capabilities; no hardcoded ids).
// Each seeded model maps to exactly one mode and fails the other three:
//   * speech     — a text-to-speech model: it ships a voice bank (audio.voices[] non-empty →
//                  Kokoro-82M), OR advertises streaming (audio.supportsStreaming → MOSS-TTS-Realtime,
//                  sc-13675), OR advertises multi-speaker dialogue (audio.supportsMultiSpeaker →
//                  MOSS-TTSD, sc-13676). A streaming / multi-speaker TTS has no fixed voice list (it
//                  speaks in its own voice(s)), so that capability is its speech signal — never a
//                  hardcoded id.
//   * music      — advertises audio-editing ops (audio.editModes[]: inpaint/repaint/extend) → ACE-Step,
//                  OR sings segmented lyrics (audio.supportsSegmentedLyrics → the six YuE lyrics2song
//                  checkpoints, sc-19383). A lyrics-to-song model has no edit surface, so the lyrics
//                  capability is its music signal; without it YuE CoT would fall through to sfx.
//   * voiceclone — conditions on a reference / speaker-identity embedding
//                  (audio.conditioning ⊇ ReferenceAudio | VoiceEmbedding). → OpenVoice V2, Chatterbox-VE.
//                  ACE-Step's conditioning is "AudioEdit" (a music-edit signal), so it does NOT match.
//                  YuE ICL's ReferenceAudio is a SONG prompt (a lyrics2song model), not a voice to clone,
//                  so a model that sings segmented lyrics never serves voiceclone.
//   * sfx        — a general text-to-audio generator (audio.sampleRates[]) that is none of the above. → MOSS.
// A model that plans a SYMBOLIC SONG (audio.supportsSymbolicSong → YuE2, sc-22998) serves none of
// the four: its request is a style + lyrics + score plan with a decoder choice, which none of these
// modes builds, and it would otherwise fall through to sfx on its sample rate alone. It is surfaced
// by its own experimental surface (sc-23000), never through an existing mode.
function audioBlock(model) {
  return model?.audio && typeof model.audio === "object" ? model.audio : null;
}

function audioHasVoices(audio) {
  return Array.isArray(audio?.voices) && audio.voices.length > 0;
}

// A streaming TTS (backend Capabilities.supports_streaming, sc-13675). Its own speech signal: a
// streaming text-to-speech model has no fixed voice bank, so it serves Speech on this flag instead.
function audioSupportsStreaming(audio) {
  return audio?.supportsStreaming === true;
}

// A multi-speaker / long-form dialogue TTS (backend Capabilities.supports_multi_speaker, sc-13676).
// Like a streaming TTS it has no fixed voice bank (it maps opaque [S1]/[S2] turn labels to its own
// voices), so the multi-speaker flag is its speech signal and reveals the segmented-script editor.
function audioSupportsMultiSpeaker(audio) {
  return audio?.supportsMultiSpeaker === true;
}

function audioHasEditModes(audio) {
  return Array.isArray(audio?.editModes) && audio.editModes.length > 0;
}

// A lyrics-to-song model (backend Capabilities.supports_segmented_lyrics — YuE, sc-19383).
function audioSingsSegmentedLyrics(audio) {
  return audio?.supportsSegmentedLyrics === true;
}

function audioGenerates(audio) {
  // A generative (text→waveform) model advertises the sample rates it emits.
  return Array.isArray(audio?.sampleRates) && audio.sampleRates.length > 0;
}

function audioHasVoiceCloneConditioning(audio) {
  const conditioning = Array.isArray(audio?.conditioning) ? audio.conditioning : [];
  return conditioning.some((kind) => VOICE_CLONE_CONDITIONING.has(String(kind).toLowerCase()));
}

function audioPlansSymbolicSong(audio) {
  return audio?.supportsSymbolicSong === true;
}

export function audioModelServesMode(model, mode) {
  const audio = audioBlock(model);
  if (!audio || audioPlansSymbolicSong(audio)) {
    return false;
  }
  if (mode === "speech") {
    return audioHasVoices(audio) || audioSupportsStreaming(audio) || audioSupportsMultiSpeaker(audio);
  }
  if (mode === "music") {
    return audioHasEditModes(audio) || audioSingsSegmentedLyrics(audio);
  }
  if (mode === "voiceclone") {
    return audioHasVoiceCloneConditioning(audio) && !audioSingsSegmentedLyrics(audio);
  }
  if (mode === "sfx") {
    return (
      audioGenerates(audio) &&
      !audioHasVoices(audio) &&
      !audioSupportsStreaming(audio) &&
      !audioSupportsMultiSpeaker(audio) &&
      !audioHasEditModes(audio) &&
      !audioSingsSegmentedLyrics(audio) &&
      !audioHasVoiceCloneConditioning(audio)
    );
  }
  return false;
}

// Usable on Audio Studio: an audio-type model that isn't Mac-blocked and serves ≥1 mode.
export function audioModelUsable(model, caps) {
  return (
    model?.type === "audio" &&
    !macModelBlock(model, caps) &&
    AUDIO_MODES.some((mode) => audioModelServesMode(model, mode))
  );
}

// Document Studio — mirrors DocumentStudio.jsx `modelSupportsInterleave` (SenseNova-U1).
export function documentModelUsable(model, caps) {
  return (
    model?.type === "image" &&
    !macModelBlock(model, caps) &&
    Array.isArray(model?.capabilities) &&
    model.capabilities.includes("interleave")
  );
}

// Reference-image → JSON caption (epic 8102, sc-8110). The vision captioner is a single,
// catalog-pinned utility model (`vision_caption_qwen3vl_8b`), so usability is "this IS the
// captioner model AND it can run here", not a capability sweep. Two gates, mirroring the
// magic-prompt model gate:
//   * macModelBlock — the active-gating Rust/MLX oracle (a no-op off Mac / in observe mode).
//   * macOnly — cross-platform as of sc-8116 (epic 8103): the catalog flips `macOnly: false` now that
//     the candle qwen3_vl vision tower (candle-llm sc-8080) is in the backend-candle graph, so this
//     branch is a no-op label and the feature lights up on Windows/Linux too. The guard is kept
//     defensively: if some future catalog entry re-sets `macOnly: true` it still hides off Mac, and
//     `caps.platform` is the API host's OS (mac_capabilities in workers.rs) — empty (`""`) pre-load
//     doesn't block, matching the no-op-pre-load convention of the macGating helpers.
export function visionCaptionModelUsable(model, caps) {
  if (model?.id !== VISION_CAPTION_MODEL_ID) {
    return false;
  }
  if (macModelBlock(model, caps)) {
    return false;
  }
  if (model?.macOnly === true) {
    const platform = caps?.platform ?? "";
    if (platform && platform !== "macos") {
      return false;
    }
  }
  return true;
}

// Photo → skeleton pose detection (Pose Library "Create" tab). Like the vision captioner this is
// a single catalog-pinned UTILITY model rather than a family, so usability is "this IS the detector
// AND it can run here", not a capability sweep — `dwpose_pose_detector` declares `capabilities: []`,
// so any capability-shaped predicate would match nothing.
//
// No `platforms` / `macOnly` branch, unlike visionCaptionModelUsable: the entry is deliberately
// unscoped because the `ort` lane runs everywhere (CoreML on macOS, CUDA/CPU off-Mac). Whether the
// SURFACE is available on a gated Mac is a separate question, answered by the screen's
// `macFeatureBlock(caps, "poseFromPhoto")` — this predicate is only about the model.
export function poseDetectModelUsable(model, caps) {
  return model?.id === POSE_DETECT_MODEL_ID && !macModelBlock(model, caps);
}

// Character Studio — mirrors CharacterStudio.jsx angle/pose predicates.
export function angleModelUsable(model, caps) {
  return !macModelBlock(model, caps) && Array.isArray(model?.ui?.viewAngles) && model.ui.viewAngles.length > 0;
}

export function poseModelUsable(model, caps) {
  return (
    !macModelBlock(model, caps) &&
    !macModelFeatureBlock(model, caps, "pose") &&
    Boolean(model?.ui?.poseLibrary)
  );
}

// Strict-control modes the selected backbone advertises (sc-8245). The single source of truth is the
// model's `ui.controlModes` — supplied by the live catalog from the manifest (the worker's
// STRICT_CONTROL_ENGINES `supported_kinds`). Canonical-ordered (pose, canny, depth) and de-duplicated so
// the picker renders deterministically regardless of manifest ordering; unknown modes are dropped (the
// worker only admits pose/canny/depth). Empty array ⇒ the backbone supports no strict control and the
// panel hides. Pure (model in → modes out), so the picker and its gate share exactly one notion of
// "supported_kinds".
export const CONTROL_MODE_ORDER = ["pose", "canny", "depth"];

export function supportedControlModes(model) {
  const declared = Array.isArray(model?.ui?.controlModes) ? model.ui.controlModes : [];
  const present = new Set(
    declared
      .filter((mode) => typeof mode === "string")
      .map((mode) => mode.trim().toLowerCase()),
  );
  return CONTROL_MODE_ORDER.filter((mode) => present.has(mode));
}

// Character Studio needs Angles OR Poses support.
export function characterModelUsable(model, caps) {
  return angleModelUsable(model, caps) || poseModelUsable(model, caps);
}

export function modelInstallComplete(model) {
  return model?.installState !== "missing" && model?.installState !== "incomplete";
}

export function generationModelsForType(models, type) {
  return (models ?? []).filter(
    (model) => model?.type === type && modelInstallComplete(model) && model?.usable !== false,
  );
}

// Does the user have ≥1 complete model usable on this screen?
// Matches what the Studios' pickers actually offer (they source from models with
// installState !== "missing"), so a screen is never gated while its picker still shows a
// model. Missing and torn/incomplete installs gate the screen; usable stale installs are reported
// as installed with updateAvailable and therefore stay available.
export function hasUsableModelFor(models, predicate, caps) {
  // `usable !== false` keeps the screen-gate in lockstep with the pickers, which
  // exclude not-yet-runnable external base models (sc-10667).
  return (models ?? []).some(
    (model) =>
      modelInstallComplete(model) &&
      model?.usable !== false &&
      predicate(model, caps),
  );
}

// The catalog entries among `ids` that a feature needs but doesn't have (epic 17625 made the
// utility resolvers — DWPose, the person detector — cache-only, so "not installed" is a job-time
// error, never a mid-job download). Returns them in the order `ids` was given, so the caller's
// declaration order is the display order.
//
// An id with NO catalog entry is treated as SATISFIED and omitted, matching the Pose Library gate:
// `models` is empty on first render, and an API too old to declare a utility entry still has a
// worker that resolves it, so "not found" must never read as "missing" — that would block a working
// install. Only an entry that positively reports missing/incomplete counts.
export function missingRequiredModels(models, ids) {
  const catalog = models ?? [];
  return (ids ?? [])
    .map((id) => catalog.find((model) => model?.id === id))
    .filter((model) => model && !modelInstallComplete(model));
}

// The models to OFFER for download when a screen is gated: catalog models usable on the
// screen that aren't installed, recommended-first. Falls back to any eligible-but-not-
// installed model so every screen has at least the models that would unlock it.
export function downloadOffersFor(models, predicate, caps) {
  const eligible = (models ?? []).filter(
    (model) =>
      model?.installState !== "installed" &&
      model?.usable !== false &&
      model?.downloadable !== false &&
      predicate(model, caps),
  );
  const recommended = eligible.filter((model) => model?.recommended === true);
  return recommended.length ? recommended : eligible;
}
