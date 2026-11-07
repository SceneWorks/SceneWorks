import { isVoiceCloneConverter, isNativeCloneGenerator, DEFAULT_MAX_SPEAKERS, GENERIC_AUDIO_PROMPT_GUIDE, usablePromptGuide } from "./audioStudioHelpers.js";
import { useMemo, useState } from "react";
import { AUDIO_MODES, audioModelServesMode, audioModelUsable, downloadOffersFor } from "../../modelEligibility.js";
import { PROMPT_REFINE_MODEL_ID } from "../../constants.js";

// Extracted from AudioStudio along its existing concern boundary.
export function useAudioModelSelection({
  saved,
  audioModels,
  mode,
  models,
  macCapabilities,
  jobs
}) {
  const [model, setModel] = useState(saved.model ?? audioModels[0]?.id ?? "");

  // Models gated on the selected tab (mirrors VideoStudio.jsx): a model "serves" a mode when its
  // audio capability block matches (audioModelServesMode). The tabs, the picker and the snap effect
  // all derive from this so the user is never trapped on a mode whose model can't serve the others.
  const modelServesMode = (item, value) => audioModelServesMode(item, value);

  const modelsForMode = (value) => {
    const serving = audioModels.filter((item) => modelServesMode(item, value));
    if (value !== "voiceclone") {
      return serving;
    }
    // Voice Clone's generate model must consume a reference clip — filter the bare speaker embedder
    // (Chatterbox-VE) out so the CTA never routes a clone to an embedder. Prefer a native clone-TTS
    // generator (single-call, sc-13412) over the OpenVoice conversion converter, so the picker default
    // (index 0) becomes the native clone whenever one is installed — the capability-gated upgrade.
    return serving
      .filter(isVoiceCloneConverter)
      .sort((a, b) => Number(isNativeCloneGenerator(b)) - Number(isNativeCloneGenerator(a)));
  };

  const modeModels = modelsForMode(mode);

  // Resolve only against models that can actually generate in this mode. A reduced catalog may
  // contain audio utility components (for example Chatterbox-VE) that must never become a prompt
  // target merely because there is no compatible generator to put in the picker.
  const selectedModel =
    modeModels.find((item) => item.id === model) ?? modeModels[0] ?? null;

  // Model-availability gate: an installed audio utility alone is not enough; at least one model must
  // serve a real Audio Studio generation mode. Offers still come from the full catalog so the user
  // can install a compatible generator.
  const modelReady = AUDIO_MODES.some((value) => modelsForMode(value).length > 0);

  // The experimental Song Lab (sc-23000) is offered when the catalog carries a symbolic-song model
  // (YuE2), installed or not — the lab owns its own opt-in, licence acceptance and install. An
  // INSTALLED one also opens the studio's availability gate, so a YuE2-only install is reachable.
  const songLabModel = useMemo(
    () => (models ?? []).find((item) => item?.type === "audio" && item?.audio?.supportsSymbolicSong === true) ?? null,
    [models],
  );

  const songLabInstalled = audioModels.some((item) => item?.audio?.supportsSymbolicSong === true);

  const modelOffers = useMemo(
    () => downloadOffersFor(models, audioModelUsable, macCapabilities),
    [models, macCapabilities],
  );

  const modelDownloadJobs = useMemo(
    () => (jobs ?? []).filter((job) => job.type === "model_download"),
    [jobs],
  );

  // Same local refinement model used by Image / Video Studio. The shared control owns the
  // missing-model download affordance; Audio Studio only resolves the catalog entry.
  const refineModel = useMemo(
    () => models.find((entry) => entry.id === PROMPT_REFINE_MODEL_ID),
    [models],
  );

  // Built-ins declare a model-specific guide. Keep third-party / legacy entries useful with an
  // audio-aware fallback instead of silently hiding the guide and refinement surfaces.
  const declaredPromptGuide = selectedModel?.ui?.promptGuide;

  const promptGuide = usablePromptGuide(declaredPromptGuide)
    ? declaredPromptGuide
    : GENERIC_AUDIO_PROMPT_GUIDE;

  // The selected model's audio capabilities — the single source every field below reads from. Never
  // hardcoded: an absent sub-block simply hides the dependent control.
  const audio = useMemo(
    () =>
      selectedModel?.audio && typeof selectedModel.audio === "object"
        ? selectedModel.audio
        : {},
    [selectedModel?.audio],
  );

  const voices = useMemo(
    () => (Array.isArray(audio.voices) ? audio.voices : []),
    [audio],
  );

  const languages = Array.isArray(audio.languages) ? audio.languages : [];

  const editModes = Array.isArray(audio.editModes) ? audio.editModes : [];

  const sampleRates = Array.isArray(audio.sampleRates) ? audio.sampleRates : [];

  const conditioning = Array.isArray(audio.conditioning) ? audio.conditioning : [];

  const maxDurationSecs = Number.isFinite(audio.maxDurationSecs) ? audio.maxDurationSecs : null;

  // Multi-speaker cap (sc-13676): READ off the selected model's audio.maxSpeakers — never hardcoded.
  // Falls back to DEFAULT_MAX_SPEAKERS only when the model advertises multi-speaker but omits the cap.
  const maxSpeakers = Number.isFinite(audio.maxSpeakers) ? audio.maxSpeakers : DEFAULT_MAX_SPEAKERS;
  return { model, setModel, modelsForMode, modeModels, selectedModel, modelReady, songLabModel, songLabInstalled, modelOffers, modelDownloadJobs, refineModel, promptGuide, audio, voices, languages, editModes, sampleRates, conditioning, maxDurationSecs, maxSpeakers };
}
