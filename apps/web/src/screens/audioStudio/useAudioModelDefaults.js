import { DEFAULT_TARGET_DURATION_SECS, defaultScript } from "./audioStudioHelpers.js";
import { useEffect } from "react";
import { AUDIO_MODES } from "../../modelEligibility.js";
import { defaultYueSections, isSegmentedSongModel } from "../../yueSong.js";

// Extracted from AudioStudio along its existing concern boundary.
export function useAudioModelDefaults({
  modeModels,
  modelsForMode,
  mode,
  setMode,
  selectedModel,
  model,
  setModel,
  audioModels,
  setVoice,
  voices,
  setLanguage,
  languages,
  setEditMode,
  editModes,
  setTargetDurationSecs,
  maxDurationSecs,
  audio,
  speakerChoices,
  setScript,
  maxSpeakers,
  setYueSections
}) {
  // Snap a reduced catalog to its first usable mode, then snap the model state to the resolved
  // generator for that mode. This keeps prompt-free audio utilities out of the picker/refiner path.
  useEffect(() => {
    if (!modeModels.length) {
      const fallbackMode = AUDIO_MODES.find((value) => modelsForMode(value).length > 0);
      if (fallbackMode && fallbackMode !== mode) {
        setMode(fallbackMode);
      }
      return;
    }
    if (selectedModel && selectedModel.id !== model) {
      setModel(selectedModel.id);
    }
    // modelServesMode / modelsForMode close over audioModels, captured below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, model, selectedModel, audioModels]);

  // Re-seed the capability-bound selections whenever the model changes, clamped to what the new
  // model actually advertises (mirrors VideoStudio's duration/resolution/fps clamp). Keeps a restored
  // snapshot value when the new model still offers it; otherwise falls back to the model's first
  // option, so a value from a previous model can never persist onto one that lacks it.
  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    setVoice((current) => (voices.some((item) => item.id === current) ? current : voices[0]?.id ?? ""));
    setLanguage((current) => (languages.includes(current) ? current : languages[0] ?? ""));
    setEditMode((current) => (editModes.includes(current) ? current : editModes[0] ?? ""));
    setTargetDurationSecs((current) => {
      if (maxDurationSecs == null) {
        return current;
      }
      const value = Number(current);
      if (Number.isFinite(value) && value > 0 && value <= maxDurationSecs) {
        return current;
      }
      return Math.min(DEFAULT_TARGET_DURATION_SECS, maxDurationSecs);
    });
    // Multi-speaker script (sc-13676): seed a starter dialogue the first time a multi-speaker model is
    // selected (an empty editor is useless), and clamp any restored/prior segments' speaker labels to
    // those the NEW model advertises — so a label from a 2-speaker model can never persist onto one
    // with a different cap. A non-multi-speaker model leaves the script untouched (it's never read).
    if (audio.supportsMultiSpeaker) {
      const allowed = new Set(speakerChoices.map((choice) => choice.value));
      const fallback = speakerChoices[0]?.value ?? "S1";
      setScript((current) => {
        if (!Array.isArray(current) || current.length === 0) {
          return defaultScript(maxSpeakers);
        }
        return current.map((segment) => ({
          ...segment,
          speaker: allowed.has(segment?.speaker) ? segment.speaker : fallback,
        }));
      });
    }
    // YuE (sc-19385): open the lyrics editor on a verse + chorus the first time a segmented-song
    // model is selected; a restored / prior song is kept.
    if (isSegmentedSongModel(selectedModel)) {
      setYueSections((current) =>
        Array.isArray(current) && current.length > 0 ? current : defaultYueSections(),
      );
    }
    // The capability arrays are derived from selectedModel; keying on its id is the intended clamp.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel?.id]);
}
