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
// SCOPE (C1 sc-13408 + C2 sc-13409): Speech (TTS) and Sound FX are both fully wired — the Generate CTA
// submits the prompt + capability-driven knobs through createAudioJob → rememberLocalGenerationJob(
// 'audio', job), surfacing the run in the audioLocalJobs stack below via the shared audio-player card
// (A5, sc-13405). Speech carries voice/language/length/seed; Sound FX (MOSS-SoundEffect v2) carries the
// prompt + length/language + the diffusion sampling knobs guidance(CFG)/steps/seed — MOSS advertises no
// voice surface, so none is sent (the shared gen-core floor would reject an explicit voice). The
// remaining modes (C3 Music, C4 Voice Clone) keep the C0 scaffold: their fields render capability-driven,
// but submit stays inert until their stories land (the Generate CTA is disabled off a wired tab rather
// than being a silent no-op).

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
// Modes whose Generate CTA is fully wired to createAudioJob (Speech C1 sc-13408, Sound FX C2 sc-13409).
// The remaining modes keep the C0 scaffold — their CTA stays disabled rather than a silent no-op — until
// their own stories land, so this set is the single guard both `canGenerate` and `submit` read.
const WIRED_MODES = new Set(["speech", "sfx"]);
// Modes that expose the diffusion sampling knobs (CFG guidance + solver steps) in the advanced panel.
// Sound FX runs the MOSS-SoundEffect flow-matching pipeline, which reads guidance/steps off the
// top-level request; Speech (Kokoro) is not a diffusion model and ignores them, so it hides them. This
// is mode-gated rather than manifest-gated because MOSS is the sole SFX model and the `audio` sub-block
// carries no per-knob range to drive it — a future non-diffusion SFX model would move this to a
// capability flag (mirrors how showVoice/showEditModes gate on their mode + a capability array).
const SAMPLING_KNOB_MODES = new Set(["sfx"]);
// Fallback target length before the model's cap is known; always clamped to maxDurationSecs.
const DEFAULT_TARGET_DURATION_SECS = 10;

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
  // Sound FX (MOSS-SoundEffect) diffusion sampling knobs — the CFG guidance scale and the solver step
  // count. Empty ⇒ the model's own default (MOSS: CFG 4.0 / 100 steps), matching the advanced hint.
  const [guidance, setGuidance] = useState(saved.guidance ?? "");
  const [steps, setSteps] = useState(saved.steps ?? "");
  const [advancedOpen, setAdvancedOpen] = useState(saved.advancedOpen ?? false);
  // Guards a Speech run in flight so a second submit (double-click / ⌘↵) can't double-enqueue
  // (mirrors VideoStudio's `submitting`). Cleared in submit's finally.
  const [submitting, setSubmitting] = useState(false);

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
  // The CFG guidance + steps advanced knobs surface only on the diffusion-audio (Sound FX) modes the
  // selected model actually runs through a sampler — the model's `supports_guidance` surface. See
  // SAMPLING_KNOB_MODES for why this is mode-gated today.
  const showSamplingKnobs = SAMPLING_KNOB_MODES.has(mode);

  // The voice picker's <optgroup> structure — derived from the selected model's voice bank, grouped
  // by accent + gender (sc-13408). Rebuilt only when the bank changes.
  const voiceGroups = useMemo(() => groupVoicesByGenderAccent(voices), [voices]);

  // Generate is guarded (never a silent no-op): a run needs a wired mode, a non-empty prompt, an
  // installed model, and no run already in flight. The empty-prompt guard is the DoD's "disable on
  // empty prompt"; the WIRED_MODES gate keeps the still-scaffold tabs (Music / Voice Clone) inert.
  const promptReady = prompt.trim().length > 0;
  const canGenerate =
    WIRED_MODES.has(mode) && modelReady && Boolean(model) && promptReady && !submitting;

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
      guidance,
      steps,
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
        // Speech (Kokoro) ships a voice bank; omitted when unset so the model falls back to af_heart.
        payload.voice = voice || undefined;
      } else if (mode === "sfx") {
        // Sound FX (MOSS-SoundEffect) — the CFG guidance scale + solver steps ride the top-level
        // request the diffusion pipeline reads. Cleared ⇒ omitted so the model uses its own default
        // (CFG 4.0 / 100 steps); the shared gen-core floor range-checks any value we do send. No
        // `voice` is sent — MOSS advertises no voice surface and the floor rejects one.
        payload.guidance = guidance === "" ? undefined : Number(guidance);
        payload.steps = steps === "" ? undefined : Number(steps);
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
              <textarea
                aria-label="Prompt"
                className="prompt-input"
                onChange={(event) => setPrompt(event.target.value)}
                placeholder={MODE_PLACEHOLDER[mode] ?? "Describe the audio…"}
                value={prompt}
              />
              {/* The shell's primary action. C1 (sc-13408) wires the real Speech submission via the
                  form's onSubmit. Disabled — never a silent no-op — until a run can proceed: an empty
                  script, no installed model, a non-Speech tab, or a run already in flight all block it. */}
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
                {/* Sound FX diffusion sampling knobs (MOSS-SoundEffect). Guidance is the CFG scale and
                    Steps is the solver step count — both ride the top-level request the flow-matching
                    pipeline reads. Cleared ⇒ the model's own default; the gen-core floor range-checks
                    any value sent, so no hardcoded ceiling is baked into the input. */}
                {showSamplingKnobs ? (
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
                {showSamplingKnobs ? (
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
