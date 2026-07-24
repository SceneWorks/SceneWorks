import { terminalStatuses } from "./jobTypes.js";
import { jobAudioResultAssets } from "./jobResultAssets.js";

// Shared, pure derivations behind the Audio Studio redesign (epic 14361). The take grid,
// the play deck and the Simple UI surfaces all read a run's mode, model, settings chips and
// relative time from these — so the Advanced screen and the Simple screen can't disagree
// about what a run was.
//
// Everything here is DERIVED, never stored: a "run" is a job in the audio local-job lane,
// its takes are the assets that job produced, and the header chips are read off the job's
// own submitted `payload`. No new API surface.

export const AUDIO_MODE_LABELS = Object.freeze({
  speech: "Speech",
  sfx: "Sound FX",
  music: "Music",
  voiceclone: "Voice Clone",
});

// m:ss clock for every transport read-out. Clamps NaN/negative to 0:00.
export function formatClock(seconds) {
  const total = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

// Coarse relative time for a run header ("just now" / "2 min ago" / "3 h ago" / "2 d ago").
// Deliberately coarse: the header is a glance surface, not a log.
export function formatRelativeTime(timestamp, nowMs = Date.now()) {
  const then = Date.parse(timestamp ?? "");
  if (!Number.isFinite(then)) {
    return "";
  }
  const seconds = Math.max(0, Math.floor((nowMs - then) / 1000));
  if (seconds < 45) {
    return "just now";
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes} min ago`;
  }
  const hours = Math.round(minutes / 60);
  if (hours < 24) {
    return `${hours} h ago`;
  }
  return `${Math.round(hours / 24)} d ago`;
}

// Which audio mode produced a run. The audio job DTO carries no `mode` field — the mode is
// a STUDIO concept, not a request one — so it is inferred from the knobs the run actually
// submitted, in the order that makes each one unambiguous:
//   reference clip ⇒ Voice Clone · music sub-block or an edit band ⇒ Music ·
//   a voice / dialogue script ⇒ Speech · otherwise the residual text→audio generator ⇒ SFX.
// The selected model's advertised capabilities break the remaining ties (a bare Speech run
// with no voice picked still reads as Speech because its model ships a voice bank).
export function audioJobMode(job, model = null) {
  const payload = job?.payload ?? {};
  if (payload.referenceAudioAssetId) {
    return "voiceclone";
  }
  if (payload.bpm != null || payload.musicalKey || payload.lyrics != null || payload.editMode) {
    return "music";
  }
  if (payload.voice || (Array.isArray(payload.script) && payload.script.length > 0)) {
    return "speech";
  }
  const audio = model?.audio && typeof model.audio === "object" ? model.audio : null;
  if (audio) {
    if (Array.isArray(audio.editModes) && audio.editModes.length > 0) {
      return "music";
    }
    if (
      (Array.isArray(audio.voices) && audio.voices.length > 0) ||
      audio.supportsMultiSpeaker
    ) {
      return "speech";
    }
  }
  return "sfx";
}

// The run's own parameters, as the short mono chips the group header shows. Read straight
// off the submitted payload so a chip can never claim a setting the run didn't carry.
export function audioRunChips(job) {
  const payload = job?.payload ?? {};
  const chips = [];
  if (payload.voice) {
    chips.push(String(payload.voice));
  }
  if (payload.language) {
    chips.push(String(payload.language));
  }
  if (payload.bpm != null && payload.bpm !== "") {
    chips.push(`${payload.bpm} BPM`);
  }
  if (payload.musicalKey) {
    chips.push(String(payload.musicalKey));
  }
  if (payload.editMode) {
    chips.push(String(payload.editMode));
  }
  const duration = Number(payload.targetDurationSecs);
  if (Number.isFinite(duration) && duration > 0) {
    chips.push(`${Math.round(duration)} s`);
  }
  if (payload.seed !== null && payload.seed !== undefined && payload.seed !== "") {
    chips.push(`seed ${payload.seed}`);
  }
  return chips;
}

// Display name for the model a run used: the catalog's label when the model is still
// installed, else the raw id the payload recorded (a run whose model was since removed
// still names what produced it).
export function audioRunModelName(job, models = []) {
  const id = job?.payload?.model ?? "";
  const match = (models ?? []).find((item) => item?.id === id);
  return match?.name ?? match?.ui?.label ?? id;
}

// True while a run is still producing — the in-flight strip's gate.
export function audioJobIsRunning(job) {
  return Boolean(job) && !terminalStatuses.has(job.status);
}

// The run stack the results zone renders: one group per job in the audio local-job lane,
// newest first, each carrying its resolved takes. In-flight and failed runs are included
// (with no takes yet) so the caller can pin them above the completed groups.
export function audioRunGroups(jobs, assets, models = []) {
  return (Array.isArray(jobs) ? jobs : []).map((job) => {
    const takes = jobAudioResultAssets(job, assets);
    const model = (models ?? []).find((item) => item?.id === job?.payload?.model) ?? null;
    const mode = audioJobMode(job, model);
    return {
      job,
      id: job.id,
      mode,
      modeLabel: AUDIO_MODE_LABELS[mode] ?? mode,
      modelName: audioRunModelName(job, models),
      chips: audioRunChips(job),
      takes,
      running: audioJobIsRunning(job),
      createdAt: job.createdAt ?? job.startedAt ?? null,
      replayable: Boolean(job.payload),
    };
  });
}

// A run reconstructed from an asset's own REPLAY RECORD (`recipe` + `normalizedSettings`,
// written by project_store) rather than from a live job. This is what populates "Latest
// audio" on a fresh launch: the local-job lane only holds the runs THIS session enqueued, so
// without it the surface is empty until you generate something — which is not what the
// design's "last N clips in this project" head describes.
//
// The replay record is deliberately narrower than a job payload (it keeps voice / language /
// requested length / seed, not the music sub-block), so an asset-derived group shows the
// chips the asset actually recorded and no more.
function assetRunPayload(asset) {
  const recipe = asset?.recipe ?? {};
  const settings = recipe.normalizedSettings ?? {};
  const payload = { model: recipe.model ?? "" };
  if (typeof recipe.prompt === "string" && recipe.prompt.trim()) {
    payload.prompt = recipe.prompt.trim();
  }
  if (settings.voice) {
    payload.voice = settings.voice;
  }
  if (settings.language) {
    payload.language = settings.language;
  }
  if (settings.targetDurationSecs != null) {
    payload.targetDurationSecs = settings.targetDurationSecs;
  }
  if (recipe.seed != null) {
    payload.seed = recipe.seed;
  }
  if (recipe.mode === "voice_clone") {
    // Enough for audioJobMode to classify it; a replay needs the reference clip the user
    // re-picks, so the group offers no Run again (see `replayable` below).
    payload.referenceAudioAssetId = asset?.lineage?.sourceAssetId ?? "";
  }
  return payload;
}

// Project-recent clips as run groups, newest first, skipping any asset a live job group
// already covers. One group per generation set (the run identity the asset records).
export function audioAssetRunGroups(assets, models = [], coveredAssetIds = new Set()) {
  const groups = [];
  const byRun = new Map();
  for (const asset of Array.isArray(assets) ? assets : []) {
    if (!asset || asset.type !== "audio" || coveredAssetIds.has(asset.id)) {
      continue;
    }
    const key = asset.generationSetId ?? `asset:${asset.id}`;
    const existing = byRun.get(key);
    if (existing) {
      existing.takes.push(asset);
      continue;
    }
    const payload = assetRunPayload(asset);
    const job = { id: `run:${key}`, status: "completed", createdAt: asset.createdAt ?? null, payload };
    const model = (models ?? []).find((item) => item?.id === payload.model) ?? null;
    const mode = audioJobMode(job, model);
    const group = {
      job,
      id: job.id,
      mode,
      modeLabel: AUDIO_MODE_LABELS[mode] ?? mode,
      modelName: audioRunModelName(job, models),
      chips: audioRunChips(job),
      takes: [asset],
      running: false,
      createdAt: asset.createdAt ?? null,
      // A voice clone can't be replayed without re-picking its reference clip, and a run
      // whose model is gone can't be replayed at all — so those groups hide "Run again"
      // rather than offering a button that would fail.
      replayable: mode !== "voiceclone" && Boolean(model),
    };
    byRun.set(key, group);
    groups.push(group);
  }
  // The groups themselves arrive newest-first (the recent-assets order), but the takes INSIDE
  // a run are numbered "Take 1..N" in the order the run produced them — so sort each run's
  // takes oldest-first or Take 1 would label the last clip rendered.
  for (const group of groups) {
    group.takes.sort((a, b) => Date.parse(a.createdAt ?? 0) - Date.parse(b.createdAt ?? 0));
  }
  return groups;
}

// Expected take count for an in-flight run — what the worker declared, when it has. Absent
// ⇒ the strip simply omits the "N takes" clause rather than guessing.
export function audioExpectedTakes(job) {
  const declared = Number(job?.result?.expectedCount);
  return Number.isFinite(declared) && declared > 0 ? declared : null;
}

// The line a take card / deck shows as its title: the prompt the run submitted, falling
// back to the asset's own display name for an imported or legacy clip.
export function audioTakeTitle(job, asset) {
  const prompt = typeof job?.payload?.prompt === "string" ? job.payload.prompt.trim() : "";
  return prompt || asset?.displayName || "Untitled clip";
}
