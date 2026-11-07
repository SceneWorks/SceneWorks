import { scriptSegmentsForSubmit } from "./audioStudioHelpers.js";

// Extracted from AudioStudio along its existing concern boundary.
export function useAudioSubmission({
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
}) {
  // "Run again" re-submits the run's OWN stored payload verbatim — the request the job
  // recorded, not the current control values (which may have moved on since). A new run
  // lands at the top of the stack with its own in-flight strip.
  async function runAgain(job) {
    const payload = job?.payload;
    if (!payload) {
      return;
    }
    const next = await createAudioJob?.(payload);
    if (next) {
      rememberLocalGenerationJob?.("audio", next);
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (submitting) {
      return;
    }
    // All four current modes are wired; WIRED_MODES remains the authoritative guard if a future
    // display-only mode is added. `canGenerate` also covers the
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
        model: selectedModel.id,
        prompt: prompt.trim(),
        language: language || undefined,
        targetDurationSecs: clampedDuration,
        seed: seed === "" ? null : Number(seed),
      };
      if (showSchedulerShift && schedulerShift !== "" && schedulerShift != null) {
        payload.schedulerShift = Number(schedulerShift);
      }
      if (mode === "speech") {
        if (showMultiSpeaker) {
          // Multi-speaker Speech (MOSS-TTSD, sc-13676): submit the segmented dialogue as
          // AudioParams.script (empty turns dropped). No `voice` — a multi-speaker model advertises no
          // fixed voice bank; it maps the per-segment [S1]/[S2] labels to its own voices. The prompt is
          // left empty; the script carries the text. The worker passes the script to the generator and
          // the gen-core floor gates it on supports_multi_speaker / maxSpeakers (never a hardcoded id).
          payload.prompt = "";
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
        payload.negativePrompt = negativePromptCfgOff ? undefined : negativePrompt.trim() || undefined;
        payload.steps = steps === "" ? undefined : Number(steps);
      } else if (songModel) {
        // YuE lyrics-to-song (sc-19385 → the sc-19384 job surface). The genre tags ARE the prompt
        // (space-joined, upstream genre.txt); the sections ride `lyrics` as `[label]\n…` blocks. Every
        // numeric knob is omitted when cleared so the model default applies. Guidance is an on/off
        // switch plus an optional scale (> 1; off ⇒ no scale, the worker sends 0.0 = no CFG).
        // No bpm / key / steps / length — the API refuses them for a segmented-song model.
        const optionalNumber = (value) => (value === "" ? undefined : Number(value));
        payload.prompt = songPrompt;
        payload.lyrics = songLyrics;
        payload.segments = optionalNumber(segments);
        payload.maxNewTokensPerSegment = optionalNumber(maxNewTokensPerSegment);
        payload.repetitionPenalty = optionalNumber(repetitionPenalty);
        payload.guidanceEnabled = Boolean(guidanceEnabled);
        if (guidanceEnabled) {
          payload.guidance = optionalNumber(songGuidance);
        }
        payload.quantTier = quantTier || undefined;
        if (showOutputLimiter) {
          payload.outputLimiter = outputLimiter || undefined;
        }
        if (songIcl && iclComplete) {
          payload.iclMode = iclMode;
          if (iclMode === "dual") {
            payload.iclVocalAssetId = iclVocalAssetId;
            payload.iclInstrumentalAssetId = iclInstrumentalAssetId;
          } else {
            payload.iclReferenceAssetId = iclReferenceAssetId;
          }
          if (audio.supportsReferenceRegion) {
            payload.iclStartSecs = optionalNumber(iclStartSecs);
            payload.iclEndSecs = optionalNumber(iclEndSecs);
          }
        }
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
          payload.negativePrompt = negativePromptCfgOff ? undefined : negativePrompt.trim() || undefined;
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
          } else if (editMode === "inpaint" || editMode === "repaint") {
            // Inpaint / repaint: a bounded interior window (seconds). Cleared bounds are omitted so the
            // worker/provider apply their own defaults (end unset ⇒ to the clip end).
            payload.editRegionStartSecs =
              editRegionStartSecs === "" ? undefined : Number(editRegionStartSecs);
            payload.editRegionEndSecs =
              editRegionEndSecs === "" ? undefined : Number(editRegionEndSecs);
          }
          // Cover (sc-13821) is a whole-clip restyle — no region (the worker's audio_edit_region
          // returns None for Cover); the source track + prompt are the whole input.
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
  return { runAgain, submit };
}
