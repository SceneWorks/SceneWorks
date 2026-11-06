import { DEFAULT_TARGET_DURATION_SECS } from "./audioStudioHelpers.js";
import { useState } from "react";

// Extracted from AudioStudio along its existing concern boundary.
export function useAudioDraft({
  saved
}) {
  const [prompt, setPrompt] = useState(saved.prompt ?? "");

  // Multi-speaker dialogue script (sc-13676): an ordered list of { speaker, text } turns, surfaced by
  // the segmented-script editor only when the selected model advertises audio.supportsMultiSpeaker.
  // Restored from the snapshot when present, else an empty array (the editor seeds a starter dialogue
  // the moment a multi-speaker model is selected — see the model-change clamp effect).
  const [script, setScript] = useState(Array.isArray(saved.script) ? saved.script : []);

  const [voice, setVoice] = useState(saved.voice ?? "");

  const [language, setLanguage] = useState(saved.language ?? "");

  const [editMode, setEditMode] = useState(saved.editMode ?? "");

  const [targetDurationSecs, setTargetDurationSecs] = useState(
    saved.targetDurationSecs ?? DEFAULT_TARGET_DURATION_SECS,
  );

  const [seed, setSeed] = useState(saved.seed ?? "");

  // Sound FX (MOSS-SoundEffect) + Music (ACE-Step) diffusion sampling knobs — the CFG guidance scale and
  // the solver step count. Empty ⇒ the model's own default (MOSS: CFG 4.0 / 100 steps; ACE-Step turbo:
  // 8 steps, guidance baked in), matching the advanced hint.
  const [guidance, setGuidance] = useState(saved.guidance ?? "");
  const [schedulerShift, setSchedulerShift] = useState(saved.schedulerShift ?? "");

  const [steps, setSteps] = useState(saved.steps ?? "");

  // Music (ACE-Step) describe-the-music sub-block — the gen-core AudioParams music fields. BPM + key are
  // optional metadata; lyrics is free-form (empty ⇒ instrumental). Cleared ⇒ omitted so the model derives
  // its own. Negative is capability-gated (hidden for the guidance-distilled turbo).
  const [bpm, setBpm] = useState(saved.bpm ?? "");

  const [musicalKey, setMusicalKey] = useState(saved.musicalKey ?? "");

  const [lyrics, setLyrics] = useState(saved.lyrics ?? "");

  const [negativePrompt, setNegativePrompt] = useState(saved.negativePrompt ?? "");

  // Music extend/edit SOURCE band (Conditioning::AudioEdit) — a USER selection, so it persists like the
  // Video Studio source band. `sourceAudioAssetId` names a library audio track; `editMode` (inpaint /
  // repaint / extend) comes from the model's advertised audio.editModes. Region seconds bound an
  // inpaint/repaint window; extend reuses the Length field as the new total. Strength is optional.
  const [sourceAudioAssetId, setSourceAudioAssetId] = useState(saved.sourceAudioAssetId ?? "");

  const [editRegionStartSecs, setEditRegionStartSecs] = useState(saved.editRegionStartSecs ?? "");

  const [editRegionEndSecs, setEditRegionEndSecs] = useState(saved.editRegionEndSecs ?? "");

  const [editStrength, setEditStrength] = useState(saved.editStrength ?? "");

  // Voice Clone (C4 sc-13411) — a USER selection, so it persists like the Music source band.
  // `referenceAudioAssetId` names the library audio track whose voice is cloned; `matchStrength` is the
  // OpenVoice conversion strength (τ), empty ⇒ the converter default (0.3).
  const [referenceAudioAssetId, setReferenceAudioAssetId] = useState(saved.referenceAudioAssetId ?? "");

  const [matchStrength, setMatchStrength] = useState(saved.matchStrength ?? "");

  // YuE lyrics-to-song (sc-19385) — separate from ACE-Step's `lyrics` so switching music models never
  // bleeds one model's controls into the other. `yueSections` is the section-labelled lyrics editor
  // ({ label, text }[]); `genreTags` is the prompt (space-joined on submit). The numeric knobs are
  // strings, empty ⇒ omitted ⇒ the model default. `outputLimiter` "" ⇒ the model default (clamp).
  const [yueSections, setYueSections] = useState(
    Array.isArray(saved.yueSections) ? saved.yueSections : [],
  );

  const [genreTags, setGenreTags] = useState(Array.isArray(saved.genreTags) ? saved.genreTags : []);

  const [segments, setSegments] = useState(saved.segments ?? "");

  const [maxNewTokensPerSegment, setMaxNewTokensPerSegment] = useState(
    saved.maxNewTokensPerSegment ?? "",
  );

  const [repetitionPenalty, setRepetitionPenalty] = useState(saved.repetitionPenalty ?? "");

  const [guidanceEnabled, setGuidanceEnabled] = useState(saved.guidanceEnabled ?? true);

  const [songGuidance, setSongGuidance] = useState(saved.songGuidance ?? "");

  const [outputLimiter, setOutputLimiter] = useState(saved.outputLimiter ?? "");

  // YuE ICL reference (sc-19385): one mixed clip ("single") or a vocal + instrumental pair ("dual"),
  // plus the window of it the prompt uses. Library audio asset ids, like the Voice Clone reference.
  const [iclMode, setIclMode] = useState(saved.iclMode ?? "single");

  const [iclReferenceAssetId, setIclReferenceAssetId] = useState(saved.iclReferenceAssetId ?? "");

  const [iclVocalAssetId, setIclVocalAssetId] = useState(saved.iclVocalAssetId ?? "");

  const [iclInstrumentalAssetId, setIclInstrumentalAssetId] = useState(
    saved.iclInstrumentalAssetId ?? "",
  );

  const [iclStartSecs, setIclStartSecs] = useState(saved.iclStartSecs ?? "");

  const [iclEndSecs, setIclEndSecs] = useState(saved.iclEndSecs ?? "");
  return { prompt, setPrompt, script, setScript, voice, setVoice, language, setLanguage, editMode, setEditMode, targetDurationSecs, setTargetDurationSecs, seed, setSeed, guidance, setGuidance, schedulerShift, setSchedulerShift, steps, setSteps, bpm, setBpm, musicalKey, setMusicalKey, lyrics, setLyrics, negativePrompt, setNegativePrompt, sourceAudioAssetId, setSourceAudioAssetId, editRegionStartSecs, setEditRegionStartSecs, editRegionEndSecs, setEditRegionEndSecs, editStrength, setEditStrength, referenceAudioAssetId, setReferenceAudioAssetId, matchStrength, setMatchStrength, yueSections, setYueSections, genreTags, setGenreTags, segments, setSegments, maxNewTokensPerSegment, setMaxNewTokensPerSegment, repetitionPenalty, setRepetitionPenalty, guidanceEnabled, setGuidanceEnabled, songGuidance, setSongGuidance, outputLimiter, setOutputLimiter, iclMode, setIclMode, iclReferenceAssetId, setIclReferenceAssetId, iclVocalAssetId, setIclVocalAssetId, iclInstrumentalAssetId, setIclInstrumentalAssetId, iclStartSecs, setIclStartSecs, iclEndSecs, setIclEndSecs };
}
