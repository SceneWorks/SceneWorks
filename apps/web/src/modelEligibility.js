// Shared per-screen model-eligibility predicates (sc-5946). Each Studio already
// derived "can this model serve this screen?" inline; those predicates are lifted here
// verbatim so the per-Studio availability gates and the screens themselves agree on
// exactly which models qualify. Predicates are capability + Mac-gating only (no install
// state) — callers layer installState via the helpers at the bottom.
import { macModelBlock, macModelFeatureBlock, macVideoModeBlock } from "./macGating.js";
import { VISION_CAPTION_MODEL_ID } from "./constants.js";

// Image generation modes a model can serve (ImageStudio mode tabs).
const IMAGE_MODES = ["text_to_image", "edit_image", "character_image", "style_variations"];

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

// Conditioning kinds that mark a voice-clone (voice-conversion / speaker-embedding) model —
// distinct from ACE-Step's "AudioEdit" conditioning, which is a music-editing signal, not a
// reference/identity signal. Compared case-insensitively so manifest casing never matters.
const VOICE_CLONE_CONDITIONING = new Set(["referenceaudio", "voiceembedding"]);

// Image Studio — mirrors ImageStudio.jsx `imageModelServesMode`. A model serves a mode
// when it declares the capability and, under active Mac gating, that feature is MLX-routed
// for it (the macModelFeatureBlock calls are no-ops off Mac).
export function imageModelServesMode(model, mode, caps) {
  const capabilities = model?.capabilities ?? [];
  if (mode === "edit_image") {
    return (
      (capabilities.includes("edit_image") || capabilities.includes("image_edit")) &&
      !macModelFeatureBlock(model, caps, "edit")
    );
  }
  if (mode === "character_image") {
    return capabilities.includes("character_image") && !macModelFeatureBlock(model, caps, "reference");
  }
  if (mode === "style_variations") {
    return capabilities.includes("style_variations") && !macModelFeatureBlock(model, caps, "reference");
  }
  return capabilities.includes("text_to_image");
}

// Usable on Image Studio: an image-type model that isn't Mac-blocked and serves ≥1 mode.
export function imageModelUsable(model, caps) {
  return (
    model?.type === "image" &&
    !macModelBlock(model, caps) &&
    IMAGE_MODES.some((mode) => imageModelServesMode(model, mode, caps))
  );
}

// Video Studio — mirrors VideoStudio.jsx `modelServesMode`.
export function videoModelServesMode(model, mode, caps) {
  return Boolean(model?.capabilities?.includes(mode)) && !macVideoModeBlock(model, caps, mode);
}

// Usable on Video Studio: a video-type model that isn't Mac-blocked and serves ≥1 mode.
export function videoModelUsable(model, caps) {
  return (
    model?.type === "video" &&
    !macModelBlock(model, caps) &&
    VIDEO_MODES.some((mode) => videoModelServesMode(model, mode, caps))
  );
}

// Audio Studio — capability-driven per-mode eligibility, derived from the model's `audio`
// sub-block (mirrors how videoModelServesMode reads video capabilities; no hardcoded ids).
// Each seeded model maps to exactly one mode and fails the other three:
//   * speech     — a text-to-speech model: it either ships a voice bank (audio.voices[] non-empty →
//                  Kokoro-82M) OR advertises streaming (audio.supportsStreaming → MOSS-TTS-Realtime,
//                  sc-13675). A streaming TTS has no fixed voice list (it speaks in its own voice), so
//                  the streaming capability is its speech signal — never a hardcoded id.
//   * music      — advertises audio-editing ops (audio.editModes[]: inpaint/repaint/extend). → ACE-Step.
//   * voiceclone — conditions on a reference / speaker-identity embedding
//                  (audio.conditioning ⊇ ReferenceAudio | VoiceEmbedding). → OpenVoice V2, Chatterbox-VE.
//                  ACE-Step's conditioning is "AudioEdit" (a music-edit signal), so it does NOT match.
//   * sfx        — a general text-to-audio generator (audio.sampleRates[]) that is none of the above. → MOSS.
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

function audioHasEditModes(audio) {
  return Array.isArray(audio?.editModes) && audio.editModes.length > 0;
}

function audioGenerates(audio) {
  // A generative (text→waveform) model advertises the sample rates it emits.
  return Array.isArray(audio?.sampleRates) && audio.sampleRates.length > 0;
}

function audioHasVoiceCloneConditioning(audio) {
  const conditioning = Array.isArray(audio?.conditioning) ? audio.conditioning : [];
  return conditioning.some((kind) => VOICE_CLONE_CONDITIONING.has(String(kind).toLowerCase()));
}

export function audioModelServesMode(model, mode) {
  const audio = audioBlock(model);
  if (!audio) {
    return false;
  }
  if (mode === "speech") {
    return audioHasVoices(audio) || audioSupportsStreaming(audio);
  }
  if (mode === "music") {
    return audioHasEditModes(audio);
  }
  if (mode === "voiceclone") {
    return audioHasVoiceCloneConditioning(audio);
  }
  if (mode === "sfx") {
    return (
      audioGenerates(audio) &&
      !audioHasVoices(audio) &&
      !audioSupportsStreaming(audio) &&
      !audioHasEditModes(audio) &&
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

// Character Studio — mirrors CharacterStudio.jsx angle/pose predicates.
export function angleModelUsable(model, caps) {
  return !macModelBlock(model, caps) && Array.isArray(model?.ui?.viewAngles) && model.ui.viewAngles.length > 0;
}

export function poseModelUsable(model, caps) {
  return !macModelBlock(model, caps) && Boolean(model?.ui?.poseLibrary);
}

// Strict-control modes the selected backbone advertises (sc-8245). The single source of truth is the
// model's `ui.controlModes` — mirrored into constants.js from the manifest (the worker's
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

// The models to OFFER for download when a screen is gated: catalog models usable on the
// screen that aren't installed, recommended-first. Falls back to any eligible-but-not-
// installed model so every screen has at least the models that would unlock it.
export function downloadOffersFor(models, predicate, caps) {
  const eligible = (models ?? []).filter(
    (model) => model?.installState !== "installed" && predicate(model, caps),
  );
  const recommended = eligible.filter((model) => model?.recommended === true);
  return recommended.length ? recommended : eligible;
}
