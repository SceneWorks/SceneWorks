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
// SCOPE (C0): the navigable shell only. The Generate CTA renders but its real submission —
// createAudioJob → rememberLocalGenerationJob('audio', job) → the results stack below — is wired by
// C1 (Speech, sc-13408) and the mode-specific stories after it. The audioLocalJobs results zone is
// already wired to the shared audio-player card (A5, sc-13405), so once C1 enqueues jobs they surface
// here with no further shell change.

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
// Fallback target length before the model's cap is known; always clamped to maxDurationSecs.
const DEFAULT_TARGET_DURATION_SECS = 10;

export function AudioStudio() {
  const {
    activeProject,
    assets = [],
    audioModels = [],
    models = [],
    jobs = [],
    audioLocalJobs = [],
    jobAction,
    createModelDownloadJob,
    setActiveView,
    setPreviewAsset,
    macCapabilities,
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
  const [voice, setVoice] = useState(saved.voice ?? "");
  const [language, setLanguage] = useState(saved.language ?? "");
  const [editMode, setEditMode] = useState(saved.editMode ?? "");
  const [targetDurationSecs, setTargetDurationSecs] = useState(
    saved.targetDurationSecs ?? DEFAULT_TARGET_DURATION_SECS,
  );
  const [seed, setSeed] = useState(saved.seed ?? "");
  const [advancedOpen, setAdvancedOpen] = useState(saved.advancedOpen ?? false);

  // Models gated on the selected tab (mirrors VideoStudio.jsx): a model "serves" a mode when its
  // audio capability block matches (audioModelServesMode). The tabs, the picker and the snap effect
  // all derive from this so the user is never trapped on a mode whose model can't serve the others.
  const modelServesMode = (item, value) => audioModelServesMode(item, value);
  const modelsForMode = (value) => audioModels.filter((item) => modelServesMode(item, value));
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

  // Which capability-driven controls the active mode surfaces. Speech leads with the full
  // voice/language/length triad C1 builds on; the other modes show a capability-driven scaffold.
  const showVoice = mode === "speech" && voices.length > 0;
  const showLanguage = DURATION_MODES.has(mode) && languages.length > 0;
  const showDuration = DURATION_MODES.has(mode) && maxDurationSecs != null;
  const showEditModes = mode === "music" && editModes.length > 0;
  const showConditioning = mode === "voiceclone" && conditioning.length > 0;

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
      voice,
      language,
      editMode,
      targetDurationSecs,
      seed,
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

  function submit(event) {
    event.preventDefault();
    // C0 is the navigable shell only. C1 (Speech, sc-13408) wires this to createAudioJob →
    // rememberLocalGenerationJob('audio', job), which surfaces the run in the audioLocalJobs stack
    // below (already rendered here via the shared audio-player card, A5 / sc-13405). The mode-specific
    // stories (C2 Sound FX, C3 Music, C4 Voice Clone) extend the same submit per mode.
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
              <textarea
                aria-label="Prompt"
                className="prompt-input"
                onChange={(event) => setPrompt(event.target.value)}
                placeholder={MODE_PLACEHOLDER[mode] ?? "Describe the audio…"}
                value={prompt}
              />
              {/* C0: the CTA renders as the shell's primary action; C1 (sc-13408) wires the real
                  submission. Kept type="submit" so the form shape matches the other studios. */}
              <button className="prompt-cta" type="submit">
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

                {/* Speech: the voice bank the selected model ships (audio.voices). C1 builds on this. */}
                {showVoice ? (
                  <label className="settings-field settings-field-voice">
                    Voice
                    <select onChange={(event) => setVoice(event.target.value)} value={voice}>
                      {voices.map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.label ?? item.id}
                        </option>
                      ))}
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
              </div>

              {/* Music editing ops the model advertises (audio.editModes) — capability-driven scaffold
                  the C3 story builds on. */}
              {showEditModes ? (
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
              ) : null}
            </div>

            {/* Voice Clone conditioning scaffold — surfaced from audio.conditioning. Reference upload +
                cloned-speech wiring is a later mode story (C4); this shows the model's capability. */}
            {showConditioning ? (
              <p className="helper-copy">
                This model conditions on a reference voice ({conditioning.join(", ")}). Reference
                upload and cloned-speech generation arrive in a later update.
              </p>
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
                {/* Sample-rate readout when the model advertises more than one; a single rate is shown
                    in the capability summary above. Capability-driven, never hardcoded. */}
                {sampleRates.length > 1 ? (
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
                <h2>Latest audio</h2>
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
