

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
export const MODE_LABELS = {
  speech: "Speech",
  sfx: "Sound FX",
  music: "Music",
  voiceclone: "Voice Clone",
};

// Per-mode prompt placeholder for each fully wired generation surface.
export const MODE_PLACEHOLDER = {
  speech: "Type the words to speak…",
  sfx: "Describe the sound — a door creak, distant thunder, footsteps on gravel…",
  music: "Describe the music — mood, instruments, tempo, genre…",
  voiceclone: "Type the words to speak in the cloned voice…",
};

// Modes that emit a fixed-length waveform, so a length control (clamped to the model's
// audio.maxDurationSecs) applies. Voice Clone follows its reference clip's length.
export const DURATION_MODES = new Set(["speech", "sfx", "music"]);

// Modes whose Generate CTA is fully wired to createAudioJob (Speech C1 sc-13408, Sound FX C2 sc-13409,
// Music C3 sc-13410, Voice Clone C4 sc-13411). This set is the single guard both `canGenerate` and
// `submit` read.
export const WIRED_MODES = new Set(["speech", "sfx", "music", "voiceclone"]);

export const audioConditioning = (item) =>
  (Array.isArray(item?.audio?.conditioning) ? item.audio.conditioning : []).map((kind) =>
    String(kind).toLowerCase(),
  );

// Voice Clone's generate model must consume a reference clip: its conditioning includes ReferenceAudio.
// Both the OpenVoice V2 conversion CONVERTER (ReferenceAudio only) and the native clone-TTS GENERATOR
// (Chatterbox chatterbox_tts, ReferenceAudio + VoiceEmbedding) qualify. A bare speaker EMBEDDER
// (Chatterbox-VE, VoiceEmbedding only) "serves" the mode conceptually but cannot render from a reference,
// so it is filtered out of the generate picker.
export function isVoiceCloneConverter(item) {
  return audioConditioning(item).includes("referenceaudio");
}

// A NATIVE clone-TTS generator (sc-13412): renders the cloned voice from script + reference in ONE call.
// The capability signature is BOTH ReferenceAudio (the clip) AND VoiceEmbedding (the speaker vector the
// generator derives from it) — the text→waveform clone generator, as opposed to the OpenVoice converter
// (ReferenceAudio only) that re-timbres a separately-synthesized base clip. When one is installed the
// Voice Clone picker prefers it (a single native step) over the two-call conversion chain.
export function isNativeCloneGenerator(item) {
  const conditioning = audioConditioning(item);
  return conditioning.includes("referenceaudio") && conditioning.includes("voiceembedding");
}

// Modes that expose the diffusion sampling knobs (CFG guidance + solver steps) in the advanced panel.
// Sound FX runs the MOSS-SoundEffect flow-matching pipeline, which reads guidance/steps off the
// top-level request; Speech (Kokoro) is not a diffusion model and ignores them, so it hides them. This
// is mode-gated rather than manifest-gated because MOSS is the sole SFX model and the `audio` sub-block
// carries no per-knob range to drive it — a future non-diffusion SFX model would move this to a
// capability flag (mirrors how showVoice/showEditModes gate on their mode + a capability array).
export const SAMPLING_KNOB_MODES = new Set(["sfx"]);

// Fallback target length before the model's cap is known; always clamped to maxDurationSecs.
export const DEFAULT_TARGET_DURATION_SECS = 10;

// Upstream YuE's `prompt_end_time` default — the ICL window end when only a start is given (the API's
// AUDIO_ICL_DEFAULT_END_SECS, sc-19384).
export const ICL_DEFAULT_END_SECS = 30;

// Multi-speaker / long-form dialogue (sc-13676). A model advertising audio.supportsMultiSpeaker
// reveals a segmented-script editor: an ordered list of turns, each { speaker, text }. The number of
// distinct speaker labels the editor offers is READ off the model's audio.maxSpeakers — never a
// hardcoded 2. When a model sets supportsMultiSpeaker but omits maxSpeakers, fall back to this so the
// editor still offers a sensible dialogue.
export const DEFAULT_MAX_SPEAKERS = 2;

// The speaker labels a multi-speaker model offers, capped at its advertised maxSpeakers. The stored
// value is the compact turn tag ("S1"/"S2") the backend maps to a voice; the display is friendlier.
export function speakerOptions(maxSpeakers) {
  const cap = Number.isFinite(maxSpeakers) && maxSpeakers >= 1 ? Math.floor(maxSpeakers) : DEFAULT_MAX_SPEAKERS;
  return Array.from({ length: cap }, (_, index) => ({
    value: `S${index + 1}`,
    label: `Speaker ${index + 1}`,
  }));
}

// The starter script for a freshly-selected multi-speaker model: one empty turn per advertised
// speaker (capped at maxSpeakers, at least one), so the editor opens ready for a two-person dialogue.
export function defaultScript(maxSpeakers) {
  return speakerOptions(maxSpeakers).map((speaker) => ({ speaker: speaker.value, text: "" }));
}

// The non-empty turns of a script, trimmed — what actually submits (empty rows are dropped). A row
// keeps its speaker label so the backend renders the right voice.
export function scriptSegmentsForSubmit(script) {
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
export const titleCase = (value) =>
  typeof value === "string" && value ? value.charAt(0).toUpperCase() + value.slice(1) : "";

// Human-readable heading for a voice group, built from whichever of accent / gender the manifest
// supplies (e.g. "American · Female"). Both absent ⇒ "" (an unlabeled bucket, rendered as bare
// options), so a model that ships a flat voice bank still renders.
export function voiceGroupLabel(accent, gender) {
  return [titleCase(accent), titleCase(gender)].filter(Boolean).join(" · ");
}

// Group a model's advertised voice bank into <optgroup>s keyed by accent + gender — the picker the
// Speech mode surfaces (sc-13408). CAPABILITY-DRIVEN: the buckets and their headings come straight
// from voices[].accent / voices[].gender, never a hardcoded list. Group order and within-group order
// follow first appearance in the advertised bank, so the picker mirrors the manifest exactly; voices
// with neither field collapse into a single unlabeled bucket rendered as bare options.
export function groupVoicesByGenderAccent(voices) {
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

export const GENERIC_AUDIO_PROMPT_GUIDE = Object.freeze({
  title: "Audio Prompt Guide",
  path: "/prompt-guides/generic-audio.md",
});

export function usablePromptGuide(guide) {
  return (
    guide &&
    typeof guide.title === "string" &&
    guide.title.trim() &&
    typeof guide.path === "string" &&
    guide.path.trim()
  );
}
