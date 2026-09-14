import React, { useEffect, useMemo, useState } from "react";
import { useAppContext } from "../context/AppContext.js";
import { audioModelServesMode } from "../modelEligibility.js";
import { audioAssetRunGroups, audioRunGroups } from "../audioTakes.js";
import { useAudioTakePlayer } from "../components/audioTakeParts.jsx";
import { buildSimpleAudioRequest } from "./simpleJobs.js";
import { SimpleAudioDeck, SimpleTakeCard, takeGridColumns } from "./simpleAudioParts.jsx";
import { useSimpleUi } from "./SimpleUiContext.js";
import { useStudioState } from "./useStudioState.js";
import { Chips, ModelDescription, SheetSelect, StudioRunStatus, jobIsRunning, newestLocalJob } from "./studioParts.jsx";

// Simple Audio Studio (design handoff → Audio Studio redesign, epic 14361 / sc-14365).
//
// Music / Speech / SFX are the three modes the design shows. Voice Clone (the fourth
// mode the advanced studio serves) is deliberately not surfaced here.
//
// The mode control is the `.su-tabs` underline tab set the Image and Video studios use, and
// the results zone is the same take grid + play deck the Advanced studio grew — sized off
// the MEASURED band (breakpointFor via the shell's `breakpoint`), never a media query. The
// old "this studio is a preview" notice is gone: the studio ships.

const MODES = [
  { id: "music", label: "Music" },
  { id: "speech", label: "Speech" },
  { id: "sfx", label: "SFX" },
];
const DURATIONS = [4, 8, 15, 30];

export function SimpleAudioStudio() {
  const {
    audioModels = [],
    createAudioJob,
    rememberLocalGenerationJob,
    audioLocalJobs = [],
    recentAudioAssets = [],
    assets = [],
    activeProject,
  } = useAppContext();
  const { toast, breakpoint } = useSimpleUi();
  // One transport for the screen, so loading a take always pauses the previous one.
  const player = useAudioTakePlayer();

  // Sticky across navigation — see the same block in SimpleImageStudio.
  const [mode, setMode] = useStudioState("audio", "mode", "music");
  const [prompt, setPrompt] = useStudioState("audio", "prompt", "");
  const [model, setModel] = useStudioState("audio", "model", "");
  const [voice, setVoice] = useStudioState("audio", "voice", "");
  const [duration, setDuration] = useStudioState("audio", "duration", 8);
  const [submitting, setSubmitting] = useState(false);

  // Per-mode eligibility comes from the model's own `audio` sub-block, exactly as the
  // advanced studio resolves it — never a hardcoded id list.
  const models = useMemo(
    () => audioModels.filter((entry) => audioModelServesMode(entry, mode)),
    [audioModels, mode],
  );
  const selectedModel = useMemo(
    () => models.find((entry) => entry.id === model) ?? null,
    [models, model],
  );

  useEffect(() => {
    if (models.length && !models.some((entry) => entry.id === model)) {
      setModel(models[0].id);
    }
  }, [models, model, setModel]);

  // The Speech voice bank the selected model declares. A streaming / multi-speaker TTS
  // ships none, in which case the Voice row is simply absent (it speaks in its own voice).
  const voices = useMemo(() => {
    const declared = selectedModel?.audio?.voices;
    return Array.isArray(declared) ? declared : [];
  }, [selectedModel]);

  useEffect(() => {
    if (voices.length && !voices.some((entry) => voiceId(entry) === voice)) {
      setVoice(voiceId(voices[0]));
    }
  }, [voices, voice, setVoice]);

  // Length is capped to the model's advertised `audio.maxDurationSecs` — never a hardcoded
  // ceiling. A model that declares no cap synthesizes its own natural length, so the control
  // is hidden and no target duration is sent (matching the advanced studio's `showDuration`).
  const maxDuration = Number.isFinite(selectedModel?.audio?.maxDurationSecs)
    ? selectedModel.audio.maxDurationSecs
    : null;
  const showDuration = maxDuration != null;
  const durations = useMemo(() => {
    if (maxDuration == null) {
      return [];
    }
    const fits = DURATIONS.filter((value) => value <= maxDuration);
    return fits.length ? fits : [maxDuration];
  }, [maxDuration]);

  useEffect(() => {
    if (durations.length && !durations.includes(duration)) {
      setDuration(durations[durations.length - 1]);
    }
  }, [durations, duration, setDuration]);

  const latestJob = newestLocalJob(audioLocalJobs);
  const busy = submitting || jobIsRunning(latestJob);
  const canGenerate = Boolean(prompt.trim()) && Boolean(model) && !busy;

  // Every take this project's audio runs have produced, newest run first — the same derived
  // grouping the Advanced studio reads, flattened into one grid (Simple shows takes, not runs).
  const runs = useMemo(() => {
    const jobRuns = audioRunGroups(audioLocalJobs, assets, audioModels);
    const covered = new Set(jobRuns.flatMap((run) => run.takes.map((asset) => asset.id)));
    return [...jobRuns, ...audioAssetRunGroups(recentAudioAssets, audioModels, covered)];
  }, [audioLocalJobs, assets, recentAudioAssets, audioModels]);
  const takes = useMemo(
    () => runs.flatMap((run) => run.takes.map((asset, index) => ({ run, asset, index }))),
    [runs],
  );
  const loaded = takes.find((take) => take.asset.id === player.loadedTakeId) ?? null;

  // "Run again" re-submits the run's OWN recorded payload, not the current controls.
  async function runAgain(job) {
    if (!job?.payload) {
      return;
    }
    const next = await createAudioJob(job.payload);
    if (next) {
      rememberLocalGenerationJob?.("audio", next);
    }
  }

  async function generate() {
    if (!canGenerate) {
      return;
    }
    if (!activeProject) {
      toast("Create or open a workspace first");
      return;
    }
    setSubmitting(true);
    try {
      const job = await createAudioJob(
        buildSimpleAudioRequest({
          model,
          prompt,
          mode,
          voice: voices.length ? voice : "",
          durationSecs: showDuration ? duration : null,
        }),
      );
      if (job) {
        rememberLocalGenerationJob?.("audio", job);
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="su-screen">
      <div className="su-tabs" role="tablist">
        {MODES.map((entry) => (
          <button
            aria-selected={mode === entry.id}
            className={mode === entry.id ? "su-tab active" : "su-tab"}
            key={entry.id}
            onClick={() => setMode(entry.id)}
            role="tab"
            type="button"
          >
            {entry.label}
          </button>
        ))}
      </div>

      <div>
        <label className="su-field-label" htmlFor="su-audio-prompt">
          {mode === "speech" ? "Script" : "Prompt"}
        </label>
        <textarea
          className="su-textarea su-textarea--short"
          id="su-audio-prompt"
          onChange={(event) => setPrompt(event.target.value)}
          placeholder={mode === "speech" ? "What should it say…" : "Describe the audio…"}
          style={{ marginTop: 6 }}
          value={prompt}
        />
      </div>

      <div className="su-settings-bar">
        <SheetSelect
          label="Model"
          onSelect={setModel}
          options={models.map((entry) => ({
            value: entry.id,
            label: entry.name ?? entry.id,
            active: entry.id === model,
          }))}
          value={selectedModel?.name ?? selectedModel?.id ?? "No audio model installed"}
        />
        {/* sc-17162 — see the note in `studioParts.jsx`. `ui.description` had one reader in the
            whole app (the advanced Models card), so Simple identified a model by NAME ALONE. */}
        <ModelDescription model={selectedModel} />
        {mode === "speech" && voices.length ? (
          <SheetSelect
            label="Voice"
            onSelect={setVoice}
            options={voices.map((entry) => ({
              value: voiceId(entry),
              label: voiceLabel(entry),
              active: voiceId(entry) === voice,
            }))}
            value={voiceLabel(voices.find((entry) => voiceId(entry) === voice)) ?? "—"}
          />
        ) : null}
        {showDuration ? (
          <Chips
            label="Duration"
            onChange={setDuration}
            options={durations.map((value) => ({ value, label: `${value}s` }))}
            value={duration}
          />
        ) : null}
      </div>

      <button
        className={busy ? "su-generate busy" : "su-generate"}
        disabled={!canGenerate}
        onClick={generate}
        type="button"
      >
        {busy ? <span aria-hidden="true" className="su-spinner" /> : null}
        {busy ? "Generating…" : "Generate audio"}
      </button>

      {/* Live run strip: progress + Cancel + the outcome, right under Generate. */}
      <StudioRunStatus job={latestJob} />

      {takes.length ? (
        <div>
          <div className="su-results-head">
            <strong>Latest audio</strong>
            <span>
              {takes.length} {takes.length === 1 ? "clip" : "clips"}
            </span>
          </div>
          {loaded ? (
            <SimpleAudioDeck
              asset={loaded.asset}
              breakpoint={breakpoint}
              onRunAgain={runAgain}
              player={player}
              run={loaded.run}
              takeIndex={loaded.index}
            />
          ) : null}
          <div className="su-take-grid" style={{ "--su-take-cols": takeGridColumns(breakpoint) }}>
            {takes.map(({ run, asset, index }) => (
              <SimpleTakeCard
                asset={asset}
                index={index}
                key={asset.id}
                loaded={asset.id === player.loadedTakeId}
                onToggle={player.toggleTake}
                phone={breakpoint === "phone"}
                playing={player.isPlaying}
                progress={player.progress}
                run={run}
              />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

// A declared voice may be a plain id string or a record; accept both so a manifest that
// grows the richer shape needs no change here.
function voiceId(entry) {
  return typeof entry === "string" ? entry : (entry?.id ?? "");
}

// The advanced studio labels a voice `item.label ?? item.id` and GROUPS the picker by
// accent/gender. This flat list can't group, so the same two fields are appended to the
// row instead — the picker stays as informative without a second grouping mechanism.
export function voiceLabel(entry) {
  if (!entry) {
    return null;
  }
  if (typeof entry === "string") {
    return entry;
  }
  const name = entry.label ?? entry.name ?? entry.id ?? "";
  const qualifiers = [entry.accent, entry.gender]
    .map((part) => (typeof part === "string" ? part.trim() : ""))
    .filter(Boolean);
  return qualifiers.length ? `${name} · ${qualifiers.join(" ")}` : name;
}
