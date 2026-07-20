import React, { useEffect, useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { AdvancedSection } from "../components/AdvancedSection.jsx";
import { WorkPanel } from "../components/WorkPanel.jsx";
import { WorkerProgressCard } from "../components/WorkerProgressCard.jsx";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import { useAppContext } from "../context/AppContext.js";
import {
  AUDIO_MODES,
  audioModelServesMode,
  audioModelUsable,
  downloadOffersFor,
} from "../modelEligibility.js";
import { loadStudioSettings, useStudioSettingsWriter } from "../hooks/useStudioSettings.js";
import { jobAudioResultAssets } from "../jobResultAssets.js";
import { AssetPickerField } from "../components/AssetPicker.jsx";

// SceneWorks Audio Studio — the navigable shell (epic 13400, C0 / sc-13407). This screen mirrors
// the canonical studio shell (VideoStudio.jsx): page-frame > WorkPanel + .mode-tabs + AdvancedSection
// + .studio-results, gated by ModelAvailabilityGate. The four modes (speech / sfx / music /
// voiceclone) come from AUDIO_MODES so the tab set and the eligibility predicates can't drift.
//
// Every settings/advanced field is CAPABILITY-DRIVEN — it reads the selected model's `audio`
// sub-block (voices / languages / editModes / sampleRates / conditioning / maxDurationSecs), never a
// hardcoded list. audioModelServesMode is the single source of truth for which model serves which
// mode (Speech = ships a voice bank, Music = advertises edit ops, Voice Clone = reference/embedding
// conditioning, Sound FX = residual text→audio generator).
//
// SCOPE (C1 sc-13408 + C2 sc-13409 + C3 sc-13410): Speech (TTS), Sound FX and Music are fully wired —
// the Generate CTA submits the prompt + capability-driven knobs through createAudioJob →
// rememberLocalGenerationJob('audio', job), surfacing the run in the audioLocalJobs stack below via the
// shared audio-player card (A5, sc-13405). Speech carries voice/language/length/seed; Sound FX
// (MOSS-SoundEffect v2) carries the prompt + length/language + the diffusion sampling knobs
// guidance(CFG)/steps/seed. Music (ACE-Step v1.5 XL Turbo) carries the describe-the-music prompt +
// length/language + BPM/key/lyrics (the gen-core AudioParams music sub-block) + steps/seed, and — ONLY
// when the selected model advertises audio.editModes — an extend/inpaint/repaint SOURCE band that picks a
// library audio track + edit mode and rides through as a Conditioning::AudioEdit (mirroring the Video
// Studio source-band pattern). guidance(CFG)/negative are capability-gated: the pinned ACE-Step turbo is
// guidance-distilled (advertises neither), so they stay hidden for it — surfacing them would be a typed
// Unsupported at the gen-core floor; a future non-distilled music model that advertises them shows them.
//
// SCOPE (C4 sc-13411 → E1 sc-13412): Voice Clone is fully wired, with TWO capability-gated backends.
// Its generate model advertises ReferenceAudio conditioning (isVoiceCloneConverter); a bare speaker
// embedder (Chatterbox-VE, VoiceEmbedding only) is filtered out of the tab/picker. When a NATIVE
// clone-TTS generator is installed — one advertising BOTH ReferenceAudio and VoiceEmbedding (Chatterbox
// chatterbox_tts, isNativeCloneGenerator) — the picker prefers it and Generate renders the cloned voice
// in a SINGLE step from the script + reference; the match-strength (τ) control is hidden because it is
// the OpenVoice converter's knob, not the native generator's. Otherwise the mode falls back to the
// OpenVoice conversion converter: it surfaces the match-strength control and the worker runs the two
// backend calls (base TTS speaks the script, then OpenVoice V2 transfers the reference's tone color).
// Either way Generate submits referenceAudioAssetId (+ matchStrength only on the conversion path) through
// createAudioJob and the worker registers the single produced clip. The native-vs-conversion routing is
// gated purely on the audio catalog's registration + Capabilities in the worker, never a hardcoded id.

// Tab labels per the epic, keyed by the AUDIO_MODES ids so the ordering follows that array.
const MODE_LABELS = {
  speech: "Speech",
  sfx: "Sound FX",
  music: "Music",
  voiceclone: "Voice Clone",
};

// Per-mode prompt placeholder. Named so the shell reads sensibly before C1 wires generation.
const MODE_PLACEHOLDER = {
  speech: "Type the words to speak…",
  sfx: "Describe the sound — a door creak, distant thunder, footsteps on gravel…",
  music: "Describe the music — mood, instruments, tempo, genre…",
  voiceclone: "Type the words to speak in the cloned voice…",
};

// Modes that emit a fixed-length waveform, so a length control (clamped to the model's
// audio.maxDurationSecs) applies. Voice Clone follows its reference clip's length.
const DURATION_MODES = new Set(["speech", "sfx", "music"]);
// Modes whose Generate CTA is fully wired to createAudioJob (Speech C1 sc-13408, Sound FX C2 sc-13409,
// Music C3 sc-13410, Voice Clone C4 sc-13411). This set is the single guard both `canGenerate` and
// `submit` read.
const WIRED_MODES = new Set(["speech", "sfx", "music", "voiceclone"]);
const audioConditioning = (item) =>
  (Array.isArray(item?.audio?.conditioning) ? item.audio.conditioning : []).map((kind) =>
    String(kind).toLowerCase(),
  );
// Voice Clone's generate model must consume a reference clip: its conditioning includes ReferenceAudio.
// Both the OpenVoice V2 conversion CONVERTER (ReferenceAudio only) and the native clone-TTS GENERATOR
// (Chatterbox chatterbox_tts, ReferenceAudio + VoiceEmbedding) qualify. A bare speaker EMBEDDER
// (Chatterbox-VE, VoiceEmbedding only) "serves" the mode conceptually but cannot render from a reference,
// so it is filtered out of the generate picker.
function isVoiceCloneConverter(item) {
  return audioConditioning(item).includes("referenceaudio");
}
// A NATIVE clone-TTS generator (sc-13412): renders the cloned voice from script + reference in ONE call.
// The capability signature is BOTH ReferenceAudio (the clip) AND VoiceEmbedding (the speaker vector the
// generator derives from it) — the text→waveform clone generator, as opposed to the OpenVoice converter
// (ReferenceAudio only) that re-timbres a separately-synthesized base clip. When one is installed the
// Voice Clone picker prefers it (a single native step) over the two-call conversion chain.
function isNativeCloneGenerator(item) {
  const conditioning = audioConditioning(item);
  return conditioning.includes("referenceaudio") && conditioning.includes("voiceembedding");
}
// Modes that expose the diffusion sampling knobs (CFG guidance + solver steps) in the advanced panel.
// Sound FX runs the MOSS-SoundEffect flow-matching pipeline, which reads guidance/steps off the
// top-level request; Speech (Kokoro) is not a diffusion model and ignores them, so it hides them. This
// is mode-gated rather than manifest-gated because MOSS is the sole SFX model and the `audio` sub-block
// carries no per-knob range to drive it — a future non-diffusion SFX model would move this to a
// capability flag (mirrors how showVoice/showEditModes gate on their mode + a capability array).
const SAMPLING_KNOB_MODES = new Set(["sfx"]);
// Fallback target length before the model's cap is known; always clamped to maxDurationSecs.
const DEFAULT_TARGET_DURATION_SECS = 10;

// Multi-speaker / long-form dialogue (sc-13676). A model advertising audio.supportsMultiSpeaker
// reveals a segmented-script editor: an ordered list of turns, each { speaker, text }. The number of
// distinct speaker labels the editor offers is READ off the model's audio.maxSpeakers — never a
// hardcoded 2. When a model sets supportsMultiSpeaker but omits maxSpeakers, fall back to this so the
// editor still offers a sensible dialogue.
const DEFAULT_MAX_SPEAKERS = 2;

// The speaker labels a multi-speaker model offers, capped at its advertised maxSpeakers. The stored
// value is the compact turn tag ("S1"/"S2") the backend maps to a voice; the display is friendlier.
function speakerOptions(maxSpeakers) {
  const cap = Number.isFinite(maxSpeakers) && maxSpeakers >= 1 ? Math.floor(maxSpeakers) : DEFAULT_MAX_SPEAKERS;
  return Array.from({ length: cap }, (_, index) => ({
    value: `S${index + 1}`,
    label: `Speaker ${index + 1}`,
  }));
}

// The starter script for a freshly-selected multi-speaker model: one empty turn per advertised
// speaker (capped at maxSpeakers, at least one), so the editor opens ready for a two-person dialogue.
function defaultScript(maxSpeakers) {
  return speakerOptions(maxSpeakers).map((speaker) => ({ speaker: speaker.value, text: "" }));
}

// The non-empty turns of a script, trimmed — what actually submits (empty rows are dropped). A row
// keeps its speaker label so the backend renders the right voice.
function scriptSegmentsForSubmit(script) {
  return (Array.isArray(script) ? script : [])
    .map((segment) => ({
      speaker: typeof segment?.speaker === "string" ? segment.speaker.trim() : "",
      text: typeof segment?.text === "string" ? segment.text.trim() : "",
    }))
    .filter((segment) => segment.text.length > 0)
    .map((segment) => (segment.speaker ? { text: segment.text, speaker: segment.speaker } : { text: segment.text }));
}

// Capitalize the first letter of a capability token for a group heading (e.g. "american" →
// "American"). The tokens come straight from the manifest, so this is display-only — never a
// hardcoded taxonomy.
const titleCase = (value) =>
  typeof value === "string" && value ? value.charAt(0).toUpperCase() + value.slice(1) : "";

// Human-readable heading for a voice group, built from whichever of accent / gender the manifest
// supplies (e.g. "American · Female"). Both absent ⇒ "" (an unlabeled bucket, rendered as bare
// options), so a model that ships a flat voice bank still renders.
function voiceGroupLabel(accent, gender) {
  return [titleCase(accent), titleCase(gender)].filter(Boolean).join(" · ");
}

// Group a model's advertised voice bank into <optgroup>s keyed by accent + gender — the picker the
// Speech mode surfaces (sc-13408). CAPABILITY-DRIVEN: the buckets and their headings come straight
// from voices[].accent / voices[].gender, never a hardcoded list. Group order and within-group order
// follow first appearance in the advertised bank, so the picker mirrors the manifest exactly; voices
// with neither field collapse into a single unlabeled bucket rendered as bare options.
function groupVoicesByGenderAccent(voices) {
  const groups = [];
  const byKey = new Map();
  for (const voice of Array.isArray(voices) ? voices : []) {
    if (!voice || typeof voice !== "object" || !voice.id) {
      continue;
    }
    const accent = typeof voice.accent === "string" ? voice.accent.trim() : "";
    const gender = typeof voice.gender === "string" ? voice.gender.trim() : "";
    const key = `${accent}|${gender}`;
    let group = byKey.get(key);
    if (!group) {
      group = { key, label: voiceGroupLabel(accent, gender), voices: [] };
      byKey.set(key, group);
      groups.push(group);
    }
    group.voices.push(voice);
  }
  return groups;
}

export function AudioStudio() {
  const {
    activeProject,
    assets = [],
    audioModels = [],
    models = [],
    jobs = [],
    audioLocalJobs = [],
    jobAction,
    createAudioJob,
    createModelDownloadJob,
    rememberLocalGenerationJob,
    setActiveView,
    setPreviewAsset,
    macCapabilities,
    savedVoices = [],
    createSavedVoice,
    deleteSavedVoice,
  } = useAppContext();

  // Last-used settings for this workspace, restored on mount. The component is keyed by workspace in
  // App.jsx, so this reads the right snapshot per workspace (mirrors Image / Video).
  const saved = useMemo(
    () => loadStudioSettings("audio", activeProject?.id ?? null),
    [activeProject?.id],
  );

  const [mode, setMode] = useState(saved.mode ?? AUDIO_MODES[0]);
  const [model, setModel] = useState(saved.model ?? audioModels[0]?.id ?? "");
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
  // Register-a-voice affordance (sc-13517): a name for the saved voice built from the currently
  // selected reference clip, an in-flight guard, and the post-register notice (dedup warning / info).
  const [savedVoiceName, setSavedVoiceName] = useState("");
  const [registeringVoice, setRegisteringVoice] = useState(false);
  const [savedVoiceNotice, setSavedVoiceNotice] = useState(null);
  const [advancedOpen, setAdvancedOpen] = useState(saved.advancedOpen ?? false);
  // Guards a Speech run in flight so a second submit (double-click / ⌘↵) can't double-enqueue
  // (mirrors VideoStudio's `submitting`). Cleared in submit's finally.
  const [submitting, setSubmitting] = useState(false);

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
  const selectedModel = audioModels.find((item) => item.id === model) ?? audioModels[0] ?? null;

  // Model-availability gate: `ready` follows the picker (audioModels is live-catalog-then-fallback in
  // App.jsx — empty only once the catalog has loaded with no installed audio model). Offers come from
  // the full catalog via audioModelUsable, recommended-first.
  const modelReady = audioModels.length > 0;
  const modelOffers = useMemo(
    () => downloadOffersFor(models, audioModelUsable, macCapabilities),
    [models, macCapabilities],
  );
  const modelDownloadJobs = useMemo(
    () => (jobs ?? []).filter((job) => job.type === "model_download"),
    [jobs],
  );

  // The selected model's audio capabilities — the single source every field below reads from. Never
  // hardcoded: an absent sub-block simply hides the dependent control.
  const audio = selectedModel?.audio && typeof selectedModel.audio === "object" ? selectedModel.audio : {};
  const voices = Array.isArray(audio.voices) ? audio.voices : [];
  const languages = Array.isArray(audio.languages) ? audio.languages : [];
  const editModes = Array.isArray(audio.editModes) ? audio.editModes : [];
  const sampleRates = Array.isArray(audio.sampleRates) ? audio.sampleRates : [];
  const conditioning = Array.isArray(audio.conditioning) ? audio.conditioning : [];
  const maxDurationSecs = Number.isFinite(audio.maxDurationSecs) ? audio.maxDurationSecs : null;
  // Multi-speaker cap (sc-13676): READ off the selected model's audio.maxSpeakers — never hardcoded.
  // Falls back to DEFAULT_MAX_SPEAKERS only when the model advertises multi-speaker but omits the cap.
  const maxSpeakers = Number.isFinite(audio.maxSpeakers) ? audio.maxSpeakers : DEFAULT_MAX_SPEAKERS;

  // Which capability-driven controls the active mode surfaces. Speech leads with the full
  // voice/language/length triad C1 builds on; the other modes show a capability-driven scaffold.
  const showVoice = mode === "speech" && voices.length > 0;
  const showLanguage = DURATION_MODES.has(mode) && languages.length > 0;
  const showDuration = DURATION_MODES.has(mode) && maxDurationSecs != null;
  // Music describe-the-music sub-block fields — surfaced on the Music tab. BPM/key/lyrics ride the
  // gen-core AudioParams music sub-block ACE-Step reads; they are optional, so they render whenever the
  // Music tab is active (a music model that ignored one would simply not consume it).
  const showMusicFields = mode === "music";
  // The extend/edit SOURCE band is revealed ONLY when the selected model advertises audio.editModes —
  // exactly the capability that makes a model a Music model (modelEligibility.audioHasEditModes). ACE-Step
  // advertises inpaint/repaint/extend; a model without editModes never shows it. Mirrors VideoStudio's
  // `studio-source-band`, which reveals its source-clip picker only on the edit sub-modes.
  const showEditModes = mode === "music" && editModes.length > 0;
  // Voice Clone: the reference-voice band (pick a library audio track) + the match-strength control are
  // revealed whenever the selected converter advertises ReferenceAudio conditioning — the capability that
  // makes it a voice-conversion model (isVoiceCloneConverter). Chatterbox-VE (VoiceEmbedding only) never
  // reaches this tab (modelsForMode filters it out).
  const showVoiceClone =
    mode === "voiceclone" && conditioning.some((kind) => String(kind).toLowerCase() === "referenceaudio");
  // Native clone-TTS (sc-13412) renders the cloned voice in a SINGLE step and has no OpenVoice
  // posterior-sampling temperature (τ), so the match-strength control is meaningful ONLY for the
  // conversion converter. Driven off the selected model's capability signature, never a hardcoded id.
  const selectedIsNativeClone = showVoiceClone && isNativeCloneGenerator(selectedModel);
  const showMatchStrength = showVoiceClone && !selectedIsNativeClone;
  // Steps: both the Sound FX (MOSS) and Music (ACE-Step) diffusion samplers read the top-level
  // `steps`, so the solver-step count surfaces on both. Guidance(CFG): only when the model advertises
  // guidance support — MOSS does (SAMPLING_KNOB_MODES), the guidance-distilled ACE-Step turbo does NOT
  // (audio.supportsGuidance falsy), so it stays hidden rather than being sent and rejected as a typed
  // Unsupported at the gen-core floor. Negative prompt is capability-gated the same way. See
  // SAMPLING_KNOB_MODES for why SFX is mode-gated while Music reads the manifest capability flags.
  const musicSupportsGuidance = mode === "music" && Boolean(audio.supportsGuidance);
  const musicSupportsNegative = mode === "music" && Boolean(audio.supportsNegativePrompt);
  const showSteps = SAMPLING_KNOB_MODES.has(mode) || mode === "music";
  const showGuidance = SAMPLING_KNOB_MODES.has(mode) || musicSupportsGuidance;
  const showNegative = musicSupportsNegative;
  // Streaming (sc-13675): the selected model renders the clip incrementally when it advertises
  // audio.supportsStreaming (backend Capabilities.supports_streaming). CAPABILITY-DRIVEN — never a
  // hardcoded id: the results zone reveals a streaming affordance and the worker's per-chunk progress
  // updates drive the WorkerProgressCard through the stream. Only a Speech model streams today
  // (MOSS-TTS-Realtime); a non-streaming model leaves it hidden so those modes are unperturbed.
  const showStreaming = Boolean(audio.supportsStreaming);
  // Multi-speaker / long-form dialogue (sc-13676): the Speech tab reveals a segmented-script editor
  // ONLY when the selected model advertises audio.supportsMultiSpeaker (backend
  // Capabilities.supports_multi_speaker). CAPABILITY-DRIVEN — never a hardcoded id: MOSS-TTSD lights
  // it up; every single-voice Speech model (Kokoro, MOSS-TTS-Realtime) leaves the plain prompt intact,
  // so those modes are unperturbed. The editor offers up to `maxSpeakers` speaker labels.
  const showMultiSpeaker = mode === "speech" && Boolean(audio.supportsMultiSpeaker);
  const speakerChoices = useMemo(() => speakerOptions(maxSpeakers), [maxSpeakers]);

  // The voice picker's <optgroup> structure — derived from the selected model's voice bank, grouped
  // by accent + gender (sc-13408). Rebuilt only when the bank changes.
  const voiceGroups = useMemo(() => groupVoicesByGenderAccent(voices), [voices]);

  // Library audio tracks the Music extend/edit source band can pick from — the audio twin of
  // VideoStudio's `videoAssets` (type-scoped so the picker only offers real audio; the picker itself
  // runs with categories hidden since its `all/image/video` tabs carry no audio bucket).
  const audioAssets = useMemo(() => assets.filter((asset) => asset?.type === "audio"), [assets]);

  // Saved voices (sc-13517): a saved voice is "selected" when the reference clip in play is the one
  // it points to. Picking a saved voice just sets referenceAudioAssetId, which the existing pipeline
  // routes to native chatterbox_tts / OpenVoice fallback — no new generation wiring.
  const selectedSavedVoiceId = useMemo(() => {
    if (!referenceAudioAssetId) return "";
    const match = savedVoices.find(
      (voice) => voice?.referenceAudioAssetId === referenceAudioAssetId,
    );
    return match?.id ?? "";
  }, [savedVoices, referenceAudioAssetId]);

  const canRegisterVoice =
    Boolean(createSavedVoice) &&
    referenceAudioAssetId.length > 0 &&
    savedVoiceName.trim().length > 0 &&
    !registeringVoice;

  async function handleRegisterVoice() {
    if (!canRegisterVoice) return;
    setRegisteringVoice(true);
    setSavedVoiceNotice(null);
    try {
      const created = await createSavedVoice({
        name: savedVoiceName.trim(),
        referenceAudioAssetId,
      });
      // A null result means the register failed — the error is surfaced by the app-level banner.
      if (!created) return;
      setSavedVoiceName("");
      const duplicate = created.nearDuplicate;
      if (duplicate) {
        setSavedVoiceNotice({
          tone: "warning",
          message: `Saved “${created.name}”, but it looks very similar to “${duplicate.name}” (${Math.round(
            duplicate.similarity * 100,
          )}% match) — you may already have this voice registered.`,
        });
      } else {
        setSavedVoiceNotice({
          tone: "info",
          message: `Saved “${created.name}” to this project’s voices.`,
        });
      }
    } finally {
      setRegisteringVoice(false);
    }
  }

  function handleSelectSavedVoice(voice) {
    setReferenceAudioAssetId(voice.referenceAudioAssetId);
    setSavedVoiceNotice(null);
  }

  async function handleDeleteSavedVoice(voice) {
    if (!deleteSavedVoice) return;
    await deleteSavedVoice(voice.id);
    setSavedVoiceNotice(null);
  }

  // Segmented-script editor handlers (sc-13676). Each edits a single { speaker, text } turn; the
  // speaker dropdown is capped at `maxSpeakers` labels so a script can never name more speakers than
  // the model renders. A row can always be added (dialogue length is unbounded — max_speakers caps
  // DISTINCT speakers, not turns) and removed down to a single turn.
  function updateSegment(index, patch) {
    setScript((current) => current.map((segment, i) => (i === index ? { ...segment, ...patch } : segment)));
  }
  function addSegment() {
    setScript((current) => {
      // Default the new turn to the speaker that keeps a two-person dialogue alternating.
      const next = speakerChoices[current.length % speakerChoices.length]?.value ?? speakerChoices[0]?.value ?? "S1";
      return [...current, { speaker: next, text: "" }];
    });
  }
  function removeSegment(index) {
    setScript((current) => (current.length > 1 ? current.filter((_, i) => i !== index) : current));
  }

  // Generate is guarded (never a silent no-op): a run needs a wired mode, generation content (a
  // non-empty prompt, or — for a multi-speaker model — at least one non-empty script turn), an
  // installed model, and no run already in flight. The empty-content guard is the DoD's "disable on
  // empty prompt"; the WIRED_MODES gate keeps the still-scaffold tabs (Music / Voice Clone) inert.
  const scriptReady = scriptSegmentsForSubmit(script).length > 0;
  // Multi-speaker Speech submits the script instead of the prompt, so its content signal is a
  // non-empty script; every other mode (and single-voice Speech) still requires a prompt.
  const contentReady = showMultiSpeaker ? scriptReady : prompt.trim().length > 0;
  // Voice Clone additionally needs a reference-voice clip selected — the conversion has no target
  // without it. The other wired modes carry no such extra requirement.
  const referenceReady = mode !== "voiceclone" || referenceAudioAssetId.length > 0;
  const canGenerate =
    WIRED_MODES.has(mode) &&
    modelReady &&
    Boolean(model) &&
    contentReady &&
    referenceReady &&
    !submitting;

  // Snap the model to one that serves the active mode (mirrors VideoStudio). A no-op when the current
  // model already serves the mode, or when nothing serves it (a reduced catalog).
  useEffect(() => {
    if (modelServesMode(selectedModel, mode)) {
      return;
    }
    const fallback = modelsForMode(mode)[0];
    if (fallback && fallback.id !== model) {
      setModel(fallback.id);
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
    // The capability arrays are derived from selectedModel; keying on its id is the intended clamp.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel?.id]);

  // Per-workspace stickiness (sc-11964 pattern). Suppressed until the audio catalog has loaded so a
  // transient default can't overwrite the restored snapshot mid-settle.
  useStudioSettingsWriter(
    "audio",
    activeProject?.id ?? null,
    {
      mode,
      model,
      prompt,
      script,
      voice,
      language,
      editMode,
      targetDurationSecs,
      seed,
      guidance,
      steps,
      bpm,
      musicalKey,
      lyrics,
      negativePrompt,
      sourceAudioAssetId,
      editRegionStartSecs,
      editRegionEndSecs,
      editStrength,
      referenceAudioAssetId,
      matchStrength,
      advancedOpen,
    },
    audioModels.length > 0,
  );

  // A human-readable capability summary (sample rate + max length) so the capability-driven nature of
  // the settings is visible at a glance and survives a model switch. Cheap enough to build inline.
  const capabilitySummary = [
    sampleRates.length
      ? sampleRates.map((rate) => `${Math.round(Number(rate) / 100) / 10} kHz`).join(" / ")
      : null,
    maxDurationSecs != null ? `up to ${maxDurationSecs}s` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  const onOpenQueue = () => setActiveView("Queue");
  const onCancelJob = (job) => jobAction?.(job, "cancel");
  const onPreview = (asset, scope) => setPreviewAsset?.(asset, scope);

  async function submit(event) {
    event.preventDefault();
    if (submitting) {
      return;
    }
    // Only the wired modes submit (Speech C1 sc-13408, Sound FX C2 sc-13409); Music / Voice Clone
    // keep the C0 scaffold, so their submit is intentionally inert. `canGenerate` also covers the
    // empty-prompt / no-model / in-flight guards, so a direct form submit can never slip past them.
    if (!canGenerate) {
      return;
    }
    setSubmitting(true);
    try {
      // Clamp the requested length to the model's advertised cap (never a hardcoded ceiling); the
      // worker range-checks it too. Omitted when the mode carries no duration control so the model
      // synthesizes its natural length.
      const durationValue = Number(targetDurationSecs);
      const clampedDuration =
        showDuration && Number.isFinite(durationValue) && durationValue > 0
          ? Math.min(durationValue, maxDurationSecs)
          : undefined;
      // The audio route deserializes `model` (not `modelId`) and reads the typed knobs verbatim
      // (apps/rust-api/src/dto.rs AudioJobRequest → crates/sceneworks-worker/src/audio_jobs.rs). The
      // language-casing seam (en-US → en-us) is handled server-side. Language is omitted when unset so
      // the model derives its own default. Mirrors how VideoStudio builds createVideoJob's payload.
      const payload = {
        model,
        prompt: prompt.trim(),
        language: language || undefined,
        targetDurationSecs: clampedDuration,
        seed: seed === "" ? null : Number(seed),
      };
      if (mode === "speech") {
        if (showMultiSpeaker) {
          // Multi-speaker Speech (MOSS-TTSD, sc-13676): submit the segmented dialogue as
          // AudioParams.script (empty turns dropped). No `voice` — a multi-speaker model advertises no
          // fixed voice bank; it maps the per-segment [S1]/[S2] labels to its own voices. The prompt is
          // left empty; the script carries the text. The worker passes the script to the generator and
          // the gen-core floor gates it on supports_multi_speaker / maxSpeakers (never a hardcoded id).
          payload.script = scriptSegmentsForSubmit(script);
        } else {
          // Single-voice Speech (Kokoro) ships a voice bank; omitted when unset so it falls back to af_heart.
          payload.voice = voice || undefined;
        }
      } else if (mode === "sfx") {
        // Sound FX (MOSS-SoundEffect) — the CFG guidance scale + solver steps ride the top-level
        // request the diffusion pipeline reads. Cleared ⇒ omitted so the model uses its own default
        // (CFG 4.0 / 100 steps); the shared gen-core floor range-checks any value we do send. No
        // `voice` is sent — MOSS advertises no voice surface and the floor rejects one.
        payload.guidance = guidance === "" ? undefined : Number(guidance);
        payload.steps = steps === "" ? undefined : Number(steps);
      } else if (mode === "music") {
        // Music (ACE-Step) — the describe-the-music sub-block. BPM/key/lyrics ride the gen-core
        // AudioParams music fields; steps rides the top-level request (the turbo's 8-step sampler).
        // Cleared ⇒ omitted so the model derives its own. guidance/negative are sent ONLY when the model
        // advertises support (the guidance-distilled turbo advertises neither, so they are never sent —
        // the shared floor would reject them as typed Unsupported).
        payload.bpm = bpm === "" ? undefined : Number(bpm);
        payload.musicalKey = musicalKey.trim() || undefined;
        payload.lyrics = lyrics.trim() || undefined;
        payload.steps = steps === "" ? undefined : Number(steps);
        if (musicSupportsGuidance) {
          payload.guidance = guidance === "" ? undefined : Number(guidance);
        }
        if (musicSupportsNegative) {
          payload.negativePrompt = negativePrompt.trim() || undefined;
        }
        // Extend/edit SOURCE band → Conditioning::AudioEdit. Only when the user has both picked a source
        // track AND chosen an edit mode the model advertises; otherwise this is plain text-to-music.
        if (showEditModes && sourceAudioAssetId && editMode) {
          payload.sourceAudioAssetId = sourceAudioAssetId;
          payload.editMode = editMode;
          if (editMode === "extend") {
            // Extend: the appended tail's new TOTAL length is the Length field; the worker defaults the
            // region start to the source clip's own length (where generation begins).
            payload.editRegionEndSecs = clampedDuration;
          } else {
            // Inpaint / repaint: a bounded interior window (seconds). Cleared bounds are omitted so the
            // worker/provider apply their own defaults (end unset ⇒ to the clip end).
            payload.editRegionStartSecs =
              editRegionStartSecs === "" ? undefined : Number(editRegionStartSecs);
            payload.editRegionEndSecs =
              editRegionEndSecs === "" ? undefined : Number(editRegionEndSecs);
          }
          payload.editStrength = editStrength === "" ? undefined : Number(editStrength);
        }
      } else if (mode === "voiceclone") {
        // Voice Clone (sc-13411 C4 → sc-13412). The prompt is the SCRIPT to speak in the cloned voice;
        // `referenceAudioAssetId` names the library audio track whose voice is cloned. `canGenerate`
        // guarantees a reference is selected. The worker routes on the selected model's capability: a
        // native clone-TTS generator renders in ONE call; otherwise the base TTS (Kokoro, `baseModel`
        // server-default) speaks and OpenVoice V2 re-timbres toward the reference. `matchStrength`
        // overrides the OpenVoice τ and is sent ONLY on the conversion path — the native clone has no τ,
        // so sending it would be a knob the model doesn't have.
        payload.referenceAudioAssetId = referenceAudioAssetId;
        if (!selectedIsNativeClone) {
          payload.matchStrength = matchStrength === "" ? undefined : Number(matchStrength);
        }
      }
      const job = await createAudioJob?.(payload);
      if (job) {
        // Land the run in the audio local-job lane so it stacks in .studio-results via the shared
        // audio-player card (A5 / sc-13405) — the audio twin of rememberLocalGenerationJob('video').
        rememberLocalGenerationJob?.("audio", job);
      }
    } finally {
      setSubmitting(false);
    }
  }

  // The model list the picker offers: those that serve the active mode, falling back to the full
  // available list if none do (a reduced catalog) so the picker is never empty.
  const pickerModels = modelsForMode(mode).length ? modelsForMode(mode) : audioModels;

  return (
    <ModelAvailabilityGate
      ready={modelReady}
      title="Audio Studio needs an audio model"
      description="Download a recommended audio model to start generating speech, music and sound."
      offers={modelOffers}
      downloadJobs={modelDownloadJobs}
      onDownload={createModelDownloadJob}
      onOpenModels={() => setActiveView("Models")}
      onOpenQueue={onOpenQueue}
      onCancelJob={onCancelJob}
    >
      <section className="page-frame audio-studio">
        <form className="studio-shell" onSubmit={submit}>
          <WorkPanel className="studio-work-panel">
            <div className="prompt-hero-top">
              <div className="mode-tabs mode-control" role="tablist" aria-label="Audio mode">
                {AUDIO_MODES.map((value) => {
                  // Disabled only when no available model serves this mode — and never the active tab,
                  // so the user can always switch away (mirrors VideoStudio's mode-level gating).
                  const blocked = value !== mode && modelsForMode(value).length === 0;
                  const active = mode === value;
                  return (
                    <button
                      className={active ? "mode-tab active" : "mode-tab"}
                      key={value}
                      role="tab"
                      aria-selected={active}
                      onClick={() => setMode(value)}
                      type="button"
                      disabled={blocked}
                      title={blocked ? "No installed model supports this mode." : undefined}
                    >
                      {MODE_LABELS[value] ?? value}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="prompt-input-row">
              {/* Multi-speaker / long-form dialogue (sc-13676): the plain prompt is replaced by a
                  segmented-script editor ONLY when the selected model advertises
                  audio.supportsMultiSpeaker. Each turn carries a speaker (capped at maxSpeakers, read
                  off the model) + its text; the script submits as AudioParams.script. Capability-driven
                  — every single-voice Speech model keeps the plain textarea, so those modes are
                  unperturbed. */}
              {showMultiSpeaker ? (
                <div
                  className="prompt-input multi-speaker-script"
                  data-testid="multi-speaker-script"
                  role="group"
                  aria-label="Multi-speaker script"
                >
                  {script.map((segment, index) => (
                    <div className="script-segment" key={index}>
                      <select
                        aria-label={`Segment ${index + 1} speaker`}
                        className="script-segment-speaker"
                        onChange={(event) => updateSegment(index, { speaker: event.target.value })}
                        value={segment.speaker ?? speakerChoices[0]?.value ?? "S1"}
                      >
                        {speakerChoices.map((choice) => (
                          <option key={choice.value} value={choice.value}>
                            {choice.label}
                          </option>
                        ))}
                      </select>
                      <textarea
                        aria-label={`Segment ${index + 1} text`}
                        className="script-segment-text"
                        onChange={(event) => updateSegment(index, { text: event.target.value })}
                        placeholder="What this speaker says…"
                        rows={2}
                        value={segment.text ?? ""}
                      />
                      <button
                        aria-label={`Remove segment ${index + 1}`}
                        className="script-segment-remove"
                        disabled={script.length <= 1}
                        onClick={() => removeSegment(index)}
                        title="Remove this turn"
                        type="button"
                      >
                        <Icon.Trash size={14} />
                      </button>
                    </div>
                  ))}
                  <button
                    className="script-add-segment"
                    data-testid="script-add-segment"
                    onClick={addSegment}
                    type="button"
                  >
                    <Icon.Plus size={14} />
                    Add turn
                  </button>
                </div>
              ) : (
                <textarea
                  aria-label="Prompt"
                  className="prompt-input"
                  onChange={(event) => setPrompt(event.target.value)}
                  placeholder={MODE_PLACEHOLDER[mode] ?? "Describe the audio…"}
                  value={prompt}
                />
              )}
              {/* The shell's primary action. C1 (sc-13408) wires the real Speech submission via the
                  form's onSubmit. Disabled — never a silent no-op — until a run can proceed: empty
                  content (prompt or multi-speaker script), no installed model, a non-Speech tab, or a
                  run already in flight all block it. */}
              <button className="prompt-cta" type="submit" disabled={!canGenerate}>
                <Icon.Sparkle size={14} />
                Generate
              </button>
            </div>

            <div className="settings-bar">
              <div className="settings-bar-row">
                <label className="settings-field settings-field-model">
                  Model
                  <select onChange={(event) => setModel(event.target.value)} value={model}>
                    {/* Models gated on the selected tab: only those that serve the active mode,
                        falling back to the full list so the picker is never empty. */}
                    {pickerModels.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name ?? item.ui?.label ?? item.id}
                      </option>
                    ))}
                  </select>
                  {capabilitySummary ? (
                    <span className="field-hint" role="note">
                      {capabilitySummary}
                    </span>
                  ) : null}
                </label>

                {/* Speech: the voice bank the selected model ships (audio.voices), grouped into
                    <optgroup>s by accent + gender (sc-13408). The buckets and their headings are
                    capability-driven — straight from voices[].accent / voices[].gender — never a
                    hardcoded taxonomy; a flat bank (no accent/gender) renders as bare options. */}
                {showVoice ? (
                  <label className="settings-field settings-field-voice">
                    Voice
                    <select onChange={(event) => setVoice(event.target.value)} value={voice}>
                      {voiceGroups.map((group) =>
                        group.label ? (
                          <optgroup key={group.key} label={group.label}>
                            {group.voices.map((item) => (
                              <option key={item.id} value={item.id}>
                                {item.label ?? item.id}
                              </option>
                            ))}
                          </optgroup>
                        ) : (
                          <React.Fragment key={group.key}>
                            {group.voices.map((item) => (
                              <option key={item.id} value={item.id}>
                                {item.label ?? item.id}
                              </option>
                            ))}
                          </React.Fragment>
                        ),
                      )}
                    </select>
                  </label>
                ) : null}

                {/* Language options the model advertises (audio.languages). */}
                {showLanguage ? (
                  <label className="settings-field settings-field-language">
                    Language
                    <select onChange={(event) => setLanguage(event.target.value)} value={language}>
                      {languages.map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}

                {/* Length, capped to the model's audio.maxDurationSecs (never a hardcoded ceiling). */}
                {showDuration ? (
                  <label className="settings-field settings-field-duration">
                    Length (s)
                    <input
                      max={maxDurationSecs}
                      min="1"
                      onChange={(event) => setTargetDurationSecs(event.target.value)}
                      step="1"
                      type="number"
                      value={targetDurationSecs}
                    />
                  </label>
                ) : null}

                {/* Music: optional tempo (BPM) + musical key. Both ride the gen-core AudioParams music
                    sub-block ACE-Step reads; cleared ⇒ omitted so the model derives its own. */}
                {showMusicFields ? (
                  <label className="settings-field settings-field-bpm">
                    BPM
                    <input
                      min="1"
                      onChange={(event) => setBpm(event.target.value)}
                      placeholder="Optional"
                      step="1"
                      type="number"
                      value={bpm}
                    />
                  </label>
                ) : null}

                {showMusicFields ? (
                  <label className="settings-field settings-field-key">
                    Key
                    <input
                      onChange={(event) => setMusicalKey(event.target.value)}
                      placeholder="e.g. C minor"
                      type="text"
                      value={musicalKey}
                    />
                  </label>
                ) : null}
              </div>

              {/* Music: optional lyrics (free-form; empty ⇒ instrumental). Rides the AudioParams
                  `lyrics` field, distinct from the describe-the-music prompt. */}
              {showMusicFields ? (
                <label className="settings-field settings-field-lyrics">
                  Lyrics
                  <textarea
                    aria-label="Lyrics"
                    onChange={(event) => setLyrics(event.target.value)}
                    placeholder="Optional — [verse] / [chorus] tags supported; leave empty for instrumental"
                    value={lyrics}
                  />
                </label>
              ) : null}

              {/* Music extend/edit SOURCE band (sc-13410) — revealed ONLY when the selected model
                  advertises audio.editModes (Conditioning::AudioEdit). Mirrors VideoStudio's
                  `studio-source-band`: pick a library audio track + an edit mode the model advertises,
                  then (for a bounded inpaint/repaint) a region window. Extend reuses the Length field as
                  the new total length. Empty source ⇒ plain text-to-music (the band is optional). */}
              {showEditModes ? (
                <div className="studio-source-band">
                  <AssetPickerField
                    assets={audioAssets}
                    buttonLabel="Select audio"
                    changeLabel="Change"
                    emptyLabel="No source track selected"
                    label="Source track (extend / edit)"
                    onChange={setSourceAudioAssetId}
                    showCategories={false}
                    value={sourceAudioAssetId}
                  />
                  <div className="settings-bar-styles">
                    <span className="settings-bar-label">Edit</span>
                    <div className="preset-chips">
                      {editModes.map((value) => (
                        <button
                          className={editMode === value ? "preset-chip active" : "preset-chip"}
                          key={value}
                          onClick={() => setEditMode(value)}
                          type="button"
                        >
                          {value}
                        </button>
                      ))}
                    </div>
                  </div>
                  {/* Inpaint / repaint bound a region (seconds) inside the source clip. Extend needs no
                      region here — its new total length is the Length field, and the worker begins the
                      appended tail at the source clip's own length. */}
                  {sourceAudioAssetId && (editMode === "inpaint" || editMode === "repaint") ? (
                    <div className="settings-bar-row">
                      <label className="settings-field settings-field-region-start">
                        Region start (s)
                        <input
                          min="0"
                          onChange={(event) => setEditRegionStartSecs(event.target.value)}
                          step="0.1"
                          type="number"
                          value={editRegionStartSecs}
                        />
                      </label>
                      <label className="settings-field settings-field-region-end">
                        Region end (s)
                        <input
                          min="0"
                          onChange={(event) => setEditRegionEndSecs(event.target.value)}
                          placeholder="To clip end"
                          step="0.1"
                          type="number"
                          value={editRegionEndSecs}
                        />
                      </label>
                    </div>
                  ) : null}
                  {/* Edit strength (0..=1) — an optional weight on the whole AudioEdit, part of the
                      Conditioning::AudioEdit contract. Cleared ⇒ the model default; a model that ignores
                      it (the ACE-Step turbo) simply falls back to its own behaviour. */}
                  {sourceAudioAssetId ? (
                    <label className="settings-field settings-field-edit-strength">
                      Edit strength
                      <input
                        max="1"
                        min="0"
                        onChange={(event) => setEditStrength(event.target.value)}
                        placeholder="Model default"
                        step="0.05"
                        type="number"
                        value={editStrength}
                      />
                    </label>
                  ) : null}
                </div>
              ) : null}
            </div>

            {/* Voice Clone (sc-13411 C4 → sc-13412): reference-voice band + (conversion-only) match
                strength. Pick a library audio track whose voice is cloned; the prompt above is the script
                spoken in that voice. A native clone-TTS generator renders it in one step; the OpenVoice
                converter re-timbres a base clip and exposes the match-strength τ. Revealed only when the
                selected model advertises ReferenceAudio conditioning (isVoiceCloneConverter). */}
            {showVoiceClone ? (
              <div className="studio-source-band">
                <AssetPickerField
                  assets={audioAssets}
                  buttonLabel="Select reference voice"
                  changeLabel="Change"
                  emptyLabel="No reference voice selected"
                  label="Reference voice"
                  onChange={setReferenceAudioAssetId}
                  showCategories={false}
                  value={referenceAudioAssetId}
                />
                {savedVoices.length > 0 ? (
                  <div className="saved-voices" role="group" aria-label="Saved voices">
                    <span className="field-label">Saved voices</span>
                    <div className="preset-chips saved-voices-chips">
                      {savedVoices.map((voice) => (
                        <span
                          key={voice.id}
                          className={`preset-chip saved-voice-chip${
                            selectedSavedVoiceId === voice.id ? " is-selected" : ""
                          }`}
                        >
                          <button
                            type="button"
                            className="saved-voice-select"
                            aria-pressed={selectedSavedVoiceId === voice.id}
                            onClick={() => handleSelectSavedVoice(voice)}
                          >
                            {voice.name}
                          </button>
                          <button
                            type="button"
                            className="saved-voice-delete"
                            aria-label={`Delete saved voice ${voice.name}`}
                            title="Delete saved voice"
                            onClick={() => handleDeleteSavedVoice(voice)}
                          >
                            <Icon.Trash />
                          </button>
                        </span>
                      ))}
                    </div>
                  </div>
                ) : null}
                {createSavedVoice ? (
                  <div className="register-voice">
                    <label className="settings-field">
                      Save this reference as a voice
                      <div className="register-voice-row">
                        <input
                          onChange={(event) => setSavedVoiceName(event.target.value)}
                          placeholder="Voice name (e.g. Narrator)"
                          type="text"
                          value={savedVoiceName}
                        />
                        <button
                          className="secondary-action register-voice-save"
                          disabled={!canRegisterVoice}
                          onClick={handleRegisterVoice}
                          type="button"
                        >
                          <Icon.Save />
                          {registeringVoice ? "Saving…" : "Save voice"}
                        </button>
                      </div>
                      <span className="field-hint" role="note">
                        Registers the selected reference clip as a reusable named voice. We compute its
                        speaker fingerprint and warn if it closely matches a voice you already saved.
                      </span>
                    </label>
                    {savedVoiceNotice ? (
                      <p
                        className={
                          savedVoiceNotice.tone === "warning" ? "inline-warning" : "field-hint"
                        }
                        role="status"
                      >
                        {savedVoiceNotice.message}
                      </p>
                    ) : null}
                  </div>
                ) : null}
                {showMatchStrength ? (
                  <label className="settings-field settings-field-match-strength">
                    Match strength
                    <input
                      max="1"
                      min="0"
                      onChange={(event) => setMatchStrength(event.target.value)}
                      placeholder="0.3 (default)"
                      step="0.05"
                      type="number"
                      value={matchStrength}
                    />
                    <span className="field-hint" role="note">
                      OpenVoice tone-color sampling temperature (τ). Cleared → model default (0.3).
                    </span>
                  </label>
                ) : null}
                <p className="helper-copy">
                  {selectedIsNativeClone
                    ? "Your script is rendered directly in the reference clip’s voice in a single step. Record or import a reference clip into the library to use it here."
                    : "The base voice speaks your script, then OpenVoice V2 converts it to match the reference clip’s voice. Record or import a reference clip into the library to use it here."}
                </p>
              </div>
            ) : null}

            <AdvancedSection
              hint="cleared values → model default"
              onToggle={() => setAdvancedOpen((value) => !value)}
              open={advancedOpen}
            >
              <div className="advanced-panel">
                <label>
                  Seed
                  <input
                    onChange={(event) => setSeed(event.target.value)}
                    placeholder="Random"
                    type="number"
                    value={seed}
                  />
                </label>
                {/* Diffusion sampling knobs. Steps (the solver step count) rides the top-level request
                    both the Sound FX (MOSS) and Music (ACE-Step) samplers read. Guidance (the CFG scale)
                    surfaces only when the model advertises guidance support — MOSS does; the
                    guidance-distilled ACE-Step turbo does NOT, so it stays hidden rather than sent (the
                    gen-core floor would reject it). Cleared ⇒ the model's own default; the floor
                    range-checks any value sent, so no hardcoded ceiling is baked into the input. */}
                {showGuidance ? (
                  <label>
                    Guidance (CFG)
                    <input
                      min="1"
                      onChange={(event) => setGuidance(event.target.value)}
                      placeholder="Model default"
                      step="0.5"
                      type="number"
                      value={guidance}
                    />
                  </label>
                ) : null}
                {showSteps ? (
                  <label>
                    Steps
                    <input
                      min="1"
                      onChange={(event) => setSteps(event.target.value)}
                      placeholder="Model default"
                      step="1"
                      type="number"
                      value={steps}
                    />
                  </label>
                ) : null}
                {/* Negative prompt — the traits to steer away from. Capability-gated: surfaced only when
                    the model advertises negative-prompt support (audio.supportsNegativePrompt). The
                    guidance-distilled ACE-Step turbo advertises none, so it stays hidden; a music model
                    that supports it renders this and rides it through as GenerationRequest.negative_prompt. */}
                {showNegative ? (
                  <label>
                    Negative prompt
                    <textarea
                      aria-label="Negative prompt"
                      onChange={(event) => setNegativePrompt(event.target.value)}
                      placeholder="Traits to avoid"
                      value={negativePrompt}
                    />
                  </label>
                ) : null}
                {/* Sample rate the model emits at (audio.sampleRates). Capability-driven, never
                    hardcoded — and read-only: the output rate is a fixed property of the model's
                    vocoder (Kokoro is 24 kHz mono), not a request knob the audio route accepts, so it
                    is surfaced for transparency rather than sent in the payload. */}
                {sampleRates.length ? (
                  <label>
                    Sample rate
                    <select disabled value={String(sampleRates[0])}>
                      {sampleRates.map((rate) => (
                        <option key={rate} value={String(rate)}>
                          {rate} Hz
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
              </div>
            </AdvancedSection>
          </WorkPanel>

          <div className="studio-results">
            <section className="review-panel">
              <div className="review-panel-head">
                <div className="review-panel-head-title">
                  <h2>Latest audio</h2>
                  {/* Streaming reveal (sc-13675): shown ONLY when the selected model advertises
                      audio.supportsStreaming. It signals that the clip renders incrementally — the
                      worker posts per-chunk progress, so the WorkerProgressCard below advances THROUGH
                      the stream as chunks arrive, and the first audio is produced before the full clip
                      finishes. Capability-driven, never a hardcoded id; hidden for every one-shot mode. */}
                  {showStreaming ? (
                    <span
                      className="streaming-badge"
                      data-testid="audio-streaming-badge"
                      title="This model streams audio incrementally — the first audio arrives before the full clip finishes rendering."
                    >
                      Streams incrementally
                    </span>
                  ) : null}
                </div>
                <span className="kbd-hint">
                  <kbd>⌘</kbd>
                  <kbd>↵</kbd>
                  to generate
                </span>
              </div>
              {/* Results zone — empty in C0. Once C1 enqueues audio jobs (via
                  rememberLocalGenerationJob('audio', job)) they surface here through the shared
                  audio-player card (A5 / sc-13405); nothing else in the shell changes. */}
              {audioLocalJobs.length ? (
                <div className="worker-progress-card-stack local-job-stack">
                  {audioLocalJobs.map((job) => {
                    const jobAssets = jobAudioResultAssets(job, assets);
                    return (
                      <WorkerProgressCard
                        key={job.id}
                        job={job}
                        thumbnailsVariant="audio-player"
                        thumbnailAssets={jobAssets}
                        onThumbnailClick={(asset) => onPreview(asset, jobAssets)}
                        onCancel={onCancelJob}
                        onOpenQueue={onOpenQueue}
                      />
                    );
                  })}
                </div>
              ) : (
                <div className="empty-panel">No audio yet</div>
              )}
            </section>
          </div>
        </form>
      </section>
    </ModelAvailabilityGate>
  );
}
