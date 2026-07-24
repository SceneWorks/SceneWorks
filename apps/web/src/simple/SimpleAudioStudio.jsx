import React, { useEffect, useMemo, useState } from "react";
import { Icon } from "../components/Icons.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
import { useAppContext } from "../context/AppContext.js";
import { audioModelServesMode } from "../modelEligibility.js";
import { resolveJobResultAssets } from "../jobResultAssets.js";
import { buildSimpleAudioRequest } from "./simpleJobs.js";
import { useSimpleUi } from "./SimpleUiContext.js";
import { Chips, DownloadButton, SheetSelect, jobIsRunning, newestLocalJob } from "./studioParts.jsx";

// Simple Audio Studio (design handoff). The design flags this studio as a PREVIEW —
// "the models and controls will change as it lands" — and asks for the warning banner
// to be kept, so it is rendered verbatim.
//
// Music / Speech / SFX are the three modes the design shows. Voice Clone (the fourth
// mode the advanced studio serves) is deliberately not surfaced here.

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
    assets = [],
    activeProject,
  } = useAppContext();
  const { toast } = useSimpleUi();

  const [mode, setMode] = useState("music");
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState("");
  const [voice, setVoice] = useState("");
  const [duration, setDuration] = useState(8);
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
  }, [models, model]);

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
  }, [voices, voice]);

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
  }, [durations, duration]);

  const latestJob = newestLocalJob(audioLocalJobs);
  const busy = submitting || jobIsRunning(latestJob);
  const canGenerate = Boolean(prompt.trim()) && Boolean(model) && !busy;
  const resultAssets = latestJob ? resolveJobResultAssets(latestJob, assets, { type: "audio" }) : [];

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
      <div className="su-segmented" role="tablist">
        {MODES.map((entry) => (
          <button
            aria-selected={mode === entry.id}
            className={mode === entry.id ? "active" : ""}
            key={entry.id}
            onClick={() => setMode(entry.id)}
            role="tab"
            type="button"
          >
            {entry.label}
          </button>
        ))}
      </div>

      <div className="su-notice" role="note">
        <Icon.Warning size={16} />
        <span>Audio Studio is a preview — the models and controls will change as it lands.</span>
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

      {resultAssets.length ? (
        <div className="su-audio-result">
          <AssetMedia asset={resultAssets[0]} />
          <DownloadButton asset={resultAssets[0]} className="su-icon-btn" />
        </div>
      ) : busy ? (
        <div className="su-audio-result">
          <span aria-hidden="true" className="su-spinner" />
          <span className="su-card-note">Rendering audio…</span>
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
