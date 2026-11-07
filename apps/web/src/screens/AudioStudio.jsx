import { AudioReviewPanel } from "./audioStudio/AudioReviewPanel.jsx";
import { AudioPromptComposer } from "./audioStudio/AudioPromptComposer.jsx";
import { AudioVoiceReferencePanel } from "./audioStudio/AudioVoiceReferencePanel.jsx";
import { AudioGenerationSettings } from "./audioStudio/AudioGenerationSettings.jsx";
import { AudioAdvancedPanel } from "./audioStudio/AudioAdvancedPanel.jsx";
import { useAudioSettingsWriter } from "./audioStudio/useAudioSettingsWriter.js";
import { useAudioSubmission } from "./audioStudio/useAudioSubmission.js";
import { useAudioModelDefaults } from "./audioStudio/useAudioModelDefaults.js";
import { useAudioScript } from "./audioStudio/useAudioScript.js";
import { useAudioVoiceLibrary } from "./audioStudio/useAudioVoiceLibrary.js";
import { useAudioDraft } from "./audioStudio/useAudioDraft.js";
import { useAudioModelSelection } from "./audioStudio/useAudioModelSelection.js";
import { MODE_LABELS, DURATION_MODES, WIRED_MODES, isNativeCloneGenerator, SAMPLING_KNOB_MODES, ICL_DEFAULT_END_SECS, speakerOptions, scriptSegmentsForSubmit, groupVoicesByGenderAccent } from "./audioStudio/audioStudioHelpers.js";
import React, { useMemo, useState } from "react";

import { Icon } from "../components/Icons.jsx";
import { AdvancedSection } from "../components/AdvancedSection.jsx";
import { WorkPanel } from "../components/WorkPanel.jsx";
import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import { useAudioTakePlayer } from "../components/audioTakeParts.jsx";

import { useAppContext } from "../context/AppContext.js";
import { AUDIO_MODES } from "../modelEligibility.js";
import { loadStudioSettings } from "../hooks/useStudioSettings.js";

import { PromptGuideModal } from "../components/PromptGuideModal.jsx";

import { useQuantTierPicker } from "../components/generationStudio.jsx";
import { installedTiers, tierPickerOptions } from "../quantTier.js";
import { YueReferenceBand } from "./audioYueControls.jsx";
import { genreTagsPrompt, isIclSongModel, isSegmentedSongModel, yueLyricsForSubmit } from "../yueSong.js";
import { Yue2SongLab } from "./audioYue2Lab.jsx";

export function AudioStudio() {
  const {
    activeProject,
    assets = [],
    audioModels = [],
    models = [],
    modelCatalogStatus = "ready",
    jobs = [],
    audioLocalJobs = [],
    recentAudioAssets = [],
    jobAction,
    createAudioJob,
    refinePrompt,
    createModelDownloadJob,
    rememberLocalGenerationJob,
    setActiveView,
    macCapabilities,
    // Take-card actions (epic 14361): the four things you can do to a clip you just heard.
    updateAssetStatus,
    deleteAsset,
    sendAssetToVideo,
    savedVoices = [],
    createSavedVoice,
    deleteSavedVoice,
    preferencesHydrated,
  } = useAppContext();

  // Last-used settings for this workspace, restored on mount. The component is keyed by workspace in
  // App.jsx, so this reads the right snapshot per workspace (mirrors Image / Video).
  const saved = useMemo(
    () => loadStudioSettings("audio", activeProject?.id ?? null),
    [activeProject?.id],
  );

  const [mode, setMode] = useState(saved.mode ?? AUDIO_MODES[0]);
  const {
    model,
    setModel,
    modelsForMode,
    modeModels,
    selectedModel,
    modelReady,
    songLabModel,
    songLabInstalled,
    modelOffers,
    modelDownloadJobs,
    refineModel,
    promptGuide,
    audio,
    voices,
    languages,
    editModes,
    sampleRates,
    conditioning,
    maxDurationSecs,
    maxSpeakers
  } = useAudioModelSelection({
    saved,
  audioModels,
  mode,
  models,
  macCapabilities,
  jobs
  });

  const {
    prompt,
    setPrompt,
    script,
    setScript,
    voice,
    setVoice,
    language,
    setLanguage,
    editMode,
    setEditMode,
    targetDurationSecs,
    setTargetDurationSecs,
    seed,
    setSeed,
    guidance,
    setGuidance,
    schedulerShift,
    setSchedulerShift,
    steps,
    setSteps,
    bpm,
    setBpm,
    musicalKey,
    setMusicalKey,
    lyrics,
    setLyrics,
    negativePrompt,
    setNegativePrompt,
    sourceAudioAssetId,
    setSourceAudioAssetId,
    editRegionStartSecs,
    setEditRegionStartSecs,
    editRegionEndSecs,
    setEditRegionEndSecs,
    editStrength,
    setEditStrength,
    referenceAudioAssetId,
    setReferenceAudioAssetId,
    matchStrength,
    setMatchStrength,
    yueSections,
    setYueSections,
    genreTags,
    setGenreTags,
    segments,
    setSegments,
    maxNewTokensPerSegment,
    setMaxNewTokensPerSegment,
    repetitionPenalty,
    setRepetitionPenalty,
    guidanceEnabled,
    setGuidanceEnabled,
    songGuidance,
    setSongGuidance,
    outputLimiter,
    setOutputLimiter,
    iclMode,
    setIclMode,
    iclReferenceAssetId,
    setIclReferenceAssetId,
    iclVocalAssetId,
    setIclVocalAssetId,
    iclInstrumentalAssetId,
    setIclInstrumentalAssetId,
    iclStartSecs,
    setIclStartSecs,
    iclEndSecs,
    setIclEndSecs
  } = useAudioDraft({
    saved
  });

  const [advancedOpen, setAdvancedOpen] = useState(saved.advancedOpen ?? false);
  // Which surface is showing (sc-23000): the standard modes, or the experimental YuE2 Song Lab.
  // The lab is entered ONLY by an explicit click on its separately-marked tab; the standard modes'
  // own `mode` state is untouched while it shows, so leaving the lab returns to exactly where the
  // user was.
  const [surface, setSurface] = useState(saved.songLab === true ? "yue2" : "standard");
  const [guideOpen, setGuideOpen] = useState(false);
  // Guards a Speech run in flight so a second submit (double-click / ⌘↵) can't double-enqueue
  // (mirrors VideoStudio's `submitting`). Cleared in submit's finally.
  const [submitting, setSubmitting] = useState(false);

  // Which capability-driven controls the active mode surfaces. Speech leads with the full
  // voice/language/length triad C1 builds on; the other modes show a capability-driven scaffold.
  // YuE lyrics-to-song (sc-19385): a Music model that sings segmented lyrics
  // (audio.supportsSegmentedLyrics) swaps ACE-Step's describe-the-music block for its own
  // section-labelled lyrics editor + genre tags + song knobs. Capability-driven, never an id match;
  // `songIcl` (an `_icl` checkpoint — ReferenceAudio conditioning) is the only model the ICL
  // reference band is enabled for. The API refuses the ACE-Step knobs for such a model
  // (validate_audio_job_for_model), so they are hidden rather than sent.
  const songModel = mode === "music" && isSegmentedSongModel(selectedModel);
  const songIcl = songModel && isIclSongModel(selectedModel);
  const showOutputLimiter = songModel && audio.supportsOutputLimiter === true;
  const showVoice = mode === "speech" && voices.length > 0;
  const showLanguage = DURATION_MODES.has(mode) && languages.length > 0;
  const showDuration = DURATION_MODES.has(mode) && maxDurationSecs != null && !songModel;
  // Music describe-the-music sub-block fields — surfaced on the Music tab. BPM/key/lyrics ride the
  // gen-core AudioParams music sub-block ACE-Step reads; they are optional, so they render whenever the
  // Music tab is active (a music model that ignored one would simply not consume it) — except for a
  // segmented-song model, which has its own lyrics editor and refuses BPM/key.
  const showMusicFields = mode === "music" && !songModel;
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
  // A segmented-song model reads no `steps` (the API refuses it) and carries guidance as its own
  // on/off + scale pair below, so both plain knobs stay ACE-Step / MOSS-only.
  const showSteps = SAMPLING_KNOB_MODES.has(mode) || (mode === "music" && !songModel);
  const showGuidance = SAMPLING_KNOB_MODES.has(mode) || (musicSupportsGuidance && !songModel);
  const showNegative = Boolean(audio.supportsNegativePrompt);
  const showSchedulerShift = Boolean(audio.supportsSchedulerShift);
  const negativePromptCfgOff = showGuidance && Number(guidance === "" ? selectedModel?.defaults?.guidanceScale : guidance) === 1;
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

  // Weight tier for a segmented-song model (YuE ships physical bf16 / q8 / q4 downloads, sc-19384):
  // the shared studio tier picker — every declared tier listed, only installed ones selectable, the
  // per-model sticky remembered under the "audio" screen. Sent as `quantTier` for a song model only.
  const songTierItems = useMemo(
    () => (songModel ? tierPickerOptions(selectedModel) : []),
    [songModel, selectedModel],
  );
  const songAvailableTiers = useMemo(
    () => (songModel ? installedTiers(selectedModel) : []),
    [songModel, selectedModel],
  );
  const {
    quantTier,
    tierSwitching,
    handleTierChange,
  } = useQuantTierPicker({
    screen: "audio",
    model: selectedModel?.id ?? "",
    selectedModel,
    availableTiers: songAvailableTiers,
    tierOptions: {},
    reseedOnModelChange: true,
    preferencesHydrated,
  });

  // The voice picker's <optgroup> structure — derived from the selected model's voice bank, grouped
  // by accent + gender (sc-13408). Rebuilt only when the bank changes.
  const voiceGroups = useMemo(() => groupVoicesByGenderAccent(voices), [voices]);

  // Library audio tracks the Music extend/edit source band can pick from — the audio twin of
  // VideoStudio's `videoAssets` (type-scoped so the picker only offers real audio; the picker itself
  // runs with categories hidden since its `all/image/video` tabs carry no audio bucket).
  const audioAssets = useMemo(() => assets.filter((asset) => asset?.type === "audio"), [assets]);

  const {
    savedVoiceName,
    setSavedVoiceName,
    registeringVoice,
    deletingVoiceId,
    savedVoiceNotice,
    selectedSavedVoiceId,
    canRegisterVoice,
    handleRegisterVoice,
    handleSelectSavedVoice,
    handleDeleteSavedVoice
  } = useAudioVoiceLibrary({
    referenceAudioAssetId,
  savedVoices,
  createSavedVoice,
  setReferenceAudioAssetId,
  deleteSavedVoice
  });

  const {
    updateSegment,
    addSegment,
    removeSegment
  } = useAudioScript({
    setScript,
  speakerChoices
  });

  // Generate is guarded (never a silent no-op): a run needs a wired mode, generation content (a
  // non-empty prompt, or — for a multi-speaker model — at least one non-empty script turn), an
  // installed model, and no run already in flight. The empty-content guard is the DoD's "disable on
  // empty prompt"; the WIRED_MODES gate keeps any future display-only mode inert.
  const scriptReady = scriptSegmentsForSubmit(script).length > 0;
  // Multi-speaker Speech submits the script instead of the prompt, so its content signal is a
  // non-empty script; every other mode (and single-voice Speech) still requires a prompt.
  // A segmented-song model needs lyrics (at least one non-empty section) AND genre tags (its prompt).
  const songLyrics = songModel ? yueLyricsForSubmit(yueSections) : "";
  const songPrompt = songModel ? genreTagsPrompt(genreTags) : "";
  const contentReady = songModel
    ? songLyrics.length > 0 && songPrompt.length > 0
    : showMultiSpeaker
      ? scriptReady
      : prompt.trim().length > 0;
  // Voice Clone additionally needs a reference-voice clip selected — the conversion has no target
  // without it. A YuE ICL checkpoint runs as a plain prompt run with NO reference (upstream allows it;
  // the API accepts it), but a half-picked dual pair is refused — so the ICL fields are sent only
  // once the chosen mode is complete, and a partial dual pick blocks Generate. The other wired modes
  // carry no such extra requirement.
  const iclPicked =
    iclMode === "dual"
      ? [iclVocalAssetId, iclInstrumentalAssetId].filter(Boolean).length
      : iclReferenceAssetId
        ? 1
        : 0;
  const iclComplete = iclMode === "dual" ? iclPicked === 2 : iclPicked === 1;
  // The reference window rides only with a complete pick on a region-capable model; a start at or
  // past the effective end (an empty end ⇒ upstream's 30 s) is refused by the API, so it blocks
  // Generate here with an inline hint instead of a 400.
  const iclWindowSent = songIcl && iclComplete && Boolean(audio.supportsReferenceRegion);
  const iclWindowError =
    iclWindowSent &&
    iclStartSecs !== "" &&
    !(Number(iclStartSecs) < (iclEndSecs === "" ? ICL_DEFAULT_END_SECS : Number(iclEndSecs)))
      ? `The reference start must be before the end (${iclEndSecs === "" ? `${ICL_DEFAULT_END_SECS} s when no end is set` : `${iclEndSecs} s`}).`
      : "";
  const iclReady = (!songIcl || iclPicked === 0 || iclComplete) && !iclWindowError;
  const referenceReady = (mode !== "voiceclone" || referenceAudioAssetId.length > 0) && iclReady;
  const canGenerate =
    WIRED_MODES.has(mode) &&
    modelReady &&
    Boolean(selectedModel) &&
    contentReady &&
    referenceReady &&
    !submitting;

  useAudioModelDefaults({
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
  });

  useAudioSettingsWriter({
    activeProject,
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
  schedulerShift,
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
  yueSections,
  genreTags,
  segments,
  maxNewTokensPerSegment,
  repetitionPenalty,
  guidanceEnabled,
  songGuidance,
  outputLimiter,
  iclMode,
  iclReferenceAssetId,
  iclVocalAssetId,
  iclInstrumentalAssetId,
  iclStartSecs,
  iclEndSecs,
  advancedOpen,
  surface,
  preferencesHydrated,
  audioModels
  });

  // A human-readable capability summary so the capability-driven nature of the settings is visible
  // at a glance and survives a model switch. Every clause is READ off the model's own audio block —
  // voice bank size, language count, output rate, length cap — and a clause a model doesn't
  // advertise is simply absent (epic 14361: the hint now owns the settings bar's footer row, so it
  // carries the whole capability picture rather than just the rate). Cheap enough to build inline.
  const capabilitySummary = [
    voices.length ? `${voices.length} ${voices.length === 1 ? "voice" : "voices"}` : null,
    languages.length ? `${languages.length} ${languages.length === 1 ? "language" : "languages"}` : null,
    sampleRates.length
      ? sampleRates.map((rate) => `${Math.round(Number(rate) / 100) / 10} kHz`).join(" / ")
      : null,
    maxDurationSecs != null ? `max ${maxDurationSecs} s` : null,
    showStreaming ? "streaming" : null,
  ]
    .filter(Boolean)
    .join(" · ");

  const onOpenQueue = () => setActiveView("Queue");
  const onCancelJob = (job) => jobAction?.(job, "cancel");

  // The results zone's now-playing transport (epic 14361). One player per screen, so loading
  // a take always pauses the previous one.
  const player = useAudioTakePlayer();

  const {
    runAgain,
    submit
  } = useAudioSubmission({
    createAudioJob,
  rememberLocalGenerationJob,
  submitting,
  canGenerate,
  setSubmitting,
  targetDurationSecs,
  showDuration,
  maxDurationSecs,
  selectedModel,
  prompt,
  language,
  seed,
  mode,
  showMultiSpeaker,
  script,
  voice,
  guidance,
  schedulerShift,
  steps,
  songModel,
  songPrompt,
  songLyrics,
  segments,
  maxNewTokensPerSegment,
  repetitionPenalty,
  guidanceEnabled,
  songGuidance,
  quantTier,
  showOutputLimiter,
  outputLimiter,
  songIcl,
  iclComplete,
  iclMode,
  iclVocalAssetId,
  iclInstrumentalAssetId,
  iclReferenceAssetId,
  audio,
  iclStartSecs,
  iclEndSecs,
  bpm,
  musicalKey,
  lyrics,
  showSchedulerShift,
  negativePromptCfgOff,
  musicSupportsGuidance,
  musicSupportsNegative,
  negativePrompt,
  showEditModes,
  sourceAudioAssetId,
  editMode,
  editRegionStartSecs,
  editRegionEndSecs,
  editStrength,
  referenceAudioAssetId,
  selectedIsNativeClone,
  matchStrength
  });

  // The picker must contain only models that can generate in the active mode. Utility-only or
  // incompatible reduced catalogs are handled by the availability gate / mode snap above.
  const pickerModels = modeModels;

  // The mode tab row, shared by the standard form and the Song Lab (sc-23000). The four standard
  // tabs are unchanged; the experimental lab entry sits in its OWN tablist after them, marked
  // experimental, so it never reads as (or displaces) a standard mode.
  const modeTabs = (
    <div className="audio-mode-tab-group">
      <div className="mode-tabs mode-control" role="tablist" aria-label="Audio mode">
        {AUDIO_MODES.map((value) => {
          // Disabled only when no available model serves this mode — and never the active tab,
          // so the user can always switch away (mirrors VideoStudio's mode-level gating).
          const blocked = value !== mode && modelsForMode(value).length === 0;
          const active = surface === "standard" && mode === value;
          return (
            <button
              className={active ? "mode-tab active" : "mode-tab"}
              key={value}
              role="tab"
              aria-selected={active}
              onClick={() => {
                setMode(value);
                setSurface("standard");
              }}
              type="button"
              disabled={blocked}
              title={blocked ? "No installed model supports this mode." : undefined}
            >
              {MODE_LABELS[value] ?? value}
            </button>
          );
        })}
      </div>
      {songLabModel ? (
        <div className="mode-tabs yue2-lab-entry" role="tablist" aria-label="Experimental audio">
          <button
            aria-selected={surface === "yue2"}
            className={surface === "yue2" ? "mode-tab active" : "mode-tab"}
            data-testid="yue2-lab-tab"
            onClick={() => setSurface("yue2")}
            role="tab"
            title="YuE2 — an experimental, noncommercial song model, separate from YuE1"
            type="button"
          >
            Song Lab <span className="yue2-experimental-badge">Experimental</span>
          </button>
        </div>
      ) : null}
    </div>
  );

  return (
    <ModelAvailabilityGate
      ready={modelReady || songLabInstalled}
      initializing={modelCatalogStatus === "idle" || modelCatalogStatus === "loading"}
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
        <form className="studio-shell" hidden={surface !== "standard"} onSubmit={submit}>
          <WorkPanel className="studio-work-panel">
            <div className="prompt-hero-top">
              {modeTabs}
              <div className="prompt-hero-links">
                <button className="hero-link" onClick={() => setGuideOpen(true)} type="button">
                  <Icon.Book size={14} /> Prompt guide
                </button>
              </div>
            </div>

            {/* sc-23000: the studio opens for a YuE2-only install (so the Song Lab is reachable), but
                the standard modes still need a standard model — keep offering the same recommended
                downloads the availability gate would, right here on the standard surface. */}
            {!modelReady ? (
              <ModelAvailabilityGate
                ready={false}
                title="No standard audio model installed"
                description="Speech, Music, Sound FX and Voice Clone need a standard audio model. The experimental Song Lab (YuE2) is separate and does not serve them."
                offers={modelOffers}
                downloadJobs={modelDownloadJobs}
                onDownload={createModelDownloadJob}
                onOpenModels={() => setActiveView("Models")}
                onOpenQueue={onOpenQueue}
                onCancelJob={onCancelJob}
              />
            ) : null}

            <AudioPromptComposer
            songModel={songModel}
            setYueSections={setYueSections}
            yueSections={yueSections}
            showMultiSpeaker={showMultiSpeaker}
            script={script}
            updateSegment={updateSegment}
            speakerChoices={speakerChoices}
            mode={mode}
            selectedModel={selectedModel}
            promptGuide={promptGuide}
            refinePrompt={refinePrompt}
            refineModel={refineModel}
            createModelDownloadJob={createModelDownloadJob}
            removeSegment={removeSegment}
            addSegment={addSegment}
            setPrompt={setPrompt}
            prompt={prompt}
            canGenerate={canGenerate}
            setGenreTags={setGenreTags}
            songPrompt={songPrompt}
          />

            {/* Settings bar (epic 14361 / sc-14363). The row is a FIXED 4-column grid — Model ·
                Voice · Language · Length — with `align-items: end`, and the model's capability
                hint has moved out of the Model label onto its own full-width footer row (shared
                with the Advanced trigger). Under the old flex row that hint lived inside the
                Model label, so `align-items: flex-end` pushed the MODEL caption ~24px above its
                neighbours and the field overflowed them. Mode-dependent extra fields keep their
                capability gating and simply become cells in the same grid. */}
            <AudioGenerationSettings
            setModel={setModel}
            selectedModel={selectedModel}
            pickerModels={pickerModels}
            showVoice={showVoice}
            setVoice={setVoice}
            voice={voice}
            voiceGroups={voiceGroups}
            showLanguage={showLanguage}
            setLanguage={setLanguage}
            language={language}
            languages={languages}
            showDuration={showDuration}
            maxDurationSecs={maxDurationSecs}
            setTargetDurationSecs={setTargetDurationSecs}
            targetDurationSecs={targetDurationSecs}
            showMusicFields={showMusicFields}
            setBpm={setBpm}
            bpm={bpm}
            setMusicalKey={setMusicalKey}
            musicalKey={musicalKey}
            songModel={songModel}
            setSegments={setSegments}
            segments={segments}
            songTierItems={songTierItems}
            handleTierChange={handleTierChange}
            tierSwitching={tierSwitching}
            quantTier={quantTier}
            setGenreTags={setGenreTags}
            genreTags={genreTags}
            setLyrics={setLyrics}
            lyrics={lyrics}
            showEditModes={showEditModes}
            audioAssets={audioAssets}
            setSourceAudioAssetId={setSourceAudioAssetId}
            sourceAudioAssetId={sourceAudioAssetId}
            editModes={editModes}
            editMode={editMode}
            setEditMode={setEditMode}
            setEditRegionStartSecs={setEditRegionStartSecs}
            editRegionStartSecs={editRegionStartSecs}
            setEditRegionEndSecs={setEditRegionEndSecs}
            editRegionEndSecs={editRegionEndSecs}
            setEditStrength={setEditStrength}
            editStrength={editStrength}
          />

            {/* Voice Clone (sc-13411 C4 → sc-13412): reference-voice band + (conversion-only) match
                strength. Pick a library audio track whose voice is cloned; the prompt above is the script
                spoken in that voice. A native clone-TTS generator renders it in one step; the OpenVoice
                converter re-timbres a base clip and exposes the match-strength τ. Revealed only when the
                selected model advertises ReferenceAudio conditioning (isVoiceCloneConverter). */}
            {/* YuE ICL reference band (sc-19385): shown for every segmented-song model so the
                capability is discoverable, but ENABLED only for an `_icl` checkpoint. */}
            {songModel ? (
              <YueReferenceBand
                audioAssets={audioAssets}
                enabled={songIcl}
                endSecs={iclEndSecs}
                iclMode={iclMode}
                instrumentalAssetId={iclInstrumentalAssetId}
                onEndChange={setIclEndSecs}
                onIclModeChange={setIclMode}
                onInstrumentalChange={setIclInstrumentalAssetId}
                onReferenceChange={setIclReferenceAssetId}
                onStartChange={setIclStartSecs}
                onVocalChange={setIclVocalAssetId}
                referenceAssetId={iclReferenceAssetId}
                showRegion={!songIcl || Boolean(audio.supportsReferenceRegion)}
                startSecs={iclStartSecs}
                vocalAssetId={iclVocalAssetId}
                windowError={iclWindowError}
              />
            ) : null}

            {showVoiceClone ? (
              <AudioVoiceReferencePanel
            audioAssets={audioAssets}
            setReferenceAudioAssetId={setReferenceAudioAssetId}
            referenceAudioAssetId={referenceAudioAssetId}
            savedVoices={savedVoices}
            selectedSavedVoiceId={selectedSavedVoiceId}
            handleSelectSavedVoice={handleSelectSavedVoice}
            deletingVoiceId={deletingVoiceId}
            handleDeleteSavedVoice={handleDeleteSavedVoice}
            createSavedVoice={createSavedVoice}
            setSavedVoiceName={setSavedVoiceName}
            savedVoiceName={savedVoiceName}
            canRegisterVoice={canRegisterVoice}
            handleRegisterVoice={handleRegisterVoice}
            registeringVoice={registeringVoice}
            savedVoiceNotice={savedVoiceNotice}
            showMatchStrength={showMatchStrength}
            setMatchStrength={setMatchStrength}
            matchStrength={matchStrength}
            selectedIsNativeClone={selectedIsNativeClone}
          />
            ) : null}

            {/* The Advanced disclosure doubles as the settings bar's footer row (screen 2a): the
                selected model's capability hint sits at its start, the trigger + caret at its end,
                and the panel opens contiguously beneath — one bordered row rather than a hint
                stranded inside the Model field. */}
            <AdvancedSection
              className="audio-advanced-section"
              hint="cleared values → model default"
              leading={
                capabilitySummary ? (
                  <span className="field-hint audio-capability-hint" role="note">
                    {capabilitySummary}
                  </span>
                ) : null
              }
              onToggle={() => setAdvancedOpen((value) => !value)}
              open={advancedOpen}
            >
              <AudioAdvancedPanel
            setSeed={setSeed}
            seed={seed}
            showGuidance={showGuidance}
            setGuidance={setGuidance}
            guidance={guidance}
            schedulerShift={schedulerShift}
            setSchedulerShift={setSchedulerShift}
            showSchedulerShift={showSchedulerShift}
            selectedModel={selectedModel}
            showSteps={showSteps}
            setSteps={setSteps}
            steps={steps}
            songModel={songModel}
            guidanceEnabled={guidanceEnabled}
            setGuidanceEnabled={setGuidanceEnabled}
            setSongGuidance={setSongGuidance}
            songGuidance={songGuidance}
            setMaxNewTokensPerSegment={setMaxNewTokensPerSegment}
            maxNewTokensPerSegment={maxNewTokensPerSegment}
            setRepetitionPenalty={setRepetitionPenalty}
            repetitionPenalty={repetitionPenalty}
            showOutputLimiter={showOutputLimiter}
            setOutputLimiter={setOutputLimiter}
            outputLimiter={outputLimiter}
            showNegative={showNegative}
            negativePromptCfgOff={negativePromptCfgOff}
            setNegativePrompt={setNegativePrompt}
            negativePrompt={negativePrompt}
            sampleRates={sampleRates}
          />
            </AdvancedSection>
          </WorkPanel>

          {/* Results zone (epic 14361 / sc-14364) — run-grouped waveform take cards, a slim
              in-flight strip and the docked play deck. The full WorkerProgressCard is no longer
              this surface's vocabulary; it stays in the Queue screen (and here only for a FAILED
              run, which needs its error + Retry actions). */}
          <AudioReviewPanel
            assets={assets}
            audioLocalJobs={audioLocalJobs}
            audioModels={audioModels}
            recentAudioAssets={recentAudioAssets}
            onCancelJob={onCancelJob}
            deleteAsset={deleteAsset}
            updateAssetStatus={updateAssetStatus}
            onOpenQueue={onOpenQueue}
            runAgain={runAgain}
            setActiveView={setActiveView}
            sendAssetToVideo={sendAssetToVideo}
            player={player}
            showStreaming={showStreaming}
          />
        </form>
        {surface === "yue2" ? (
          <Yue2SongLab header={<div className="prompt-hero-top">{modeTabs}</div>} />
        ) : null}
        {guideOpen ? (
          <PromptGuideModal
            guide={promptGuide}
            modelName={selectedModel?.name}
            onClose={() => setGuideOpen(false)}
          />
        ) : null}
      </section>
    </ModelAvailabilityGate>
  );
}
