import { terminalStatuses } from "./jobTypes.js";

export function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) {
    return "0s";
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return minutes > 0 ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

// Frame-accurate NLE timecode `MM:SS:FF` (Video Editor redesign, epic 12798). The
// third field is the frame within the current second at the timeline's fps, so the
// toolbar/ruler read like a pro editor. `fps` falls back to 30 for a nullish/invalid
// rate; seconds are clamped to >= 0. Frames are floored and clamped to fps-1 so a
// value sitting exactly on a second boundary never reads as ":30" at 30fps.
export function formatTimecode(seconds, fps = 30) {
  const safeFps = Number.isFinite(fps) && fps > 0 ? Math.round(fps) : 30;
  const total = Math.max(0, Number(seconds) || 0);
  const whole = Math.floor(total);
  const minutes = Math.floor(whole / 60);
  const secs = whole % 60;
  const frames = Math.min(safeFps - 1, Math.floor((total - whole) * safeFps));
  const pad = (value) => String(value).padStart(2, "0");
  return `${pad(minutes)}:${pad(secs)}:${pad(frames)}`;
}

export function percent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

// Human duration from milliseconds (epic 10402, generation stats). Sub-second
// reads in ms, then seconds with one decimal, then minutes. "—" when absent.
export function formatMs(ms) {
  if (ms === null || ms === undefined || !Number.isFinite(Number(ms))) {
    return "—";
  }
  const value = Number(ms);
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  const seconds = value / 1000;
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${String(remainder).padStart(2, "0")}s`;
}

// Human byte size (epic 10402), binary units. "—" when absent, "0 B" for zero.
export function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || !Number.isFinite(Number(bytes))) {
    return "—";
  }
  const value = Number(bytes);
  if (value <= 0) {
    return "0 B";
  }
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const exp = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  const scaled = value / 1024 ** exp;
  return `${scaled.toFixed(exp === 0 ? 0 : 1)} ${units[exp]}`;
}

// Percent from a 0..100 value (epic 10402), rounded, with a suffix. Distinct from
// `percent()` above, which scales a 0..1 fraction.
export function formatPercent(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return "—";
  }
  return `${Math.round(Number(value))}%`;
}

// Quant display label (epic 10402). Passes through the worker's normalized label
// (bf16 / q8 / q4 / int8-convrot), falling back to a dash when absent.
export function quantLabel(label) {
  return label && String(label).trim() ? String(label) : "—";
}

export function liveElapsedSeconds(job, nowMs = Date.now()) {
  if (terminalStatuses.has(job.status) || !job.startedAt) {
    return job.elapsedSeconds;
  }
  const startedMs = Date.parse(job.startedAt);
  if (!Number.isFinite(startedMs)) {
    return job.elapsedSeconds;
  }
  const elapsed = Math.max(0, Math.floor((nowMs - startedMs) / 1000));
  return Math.max(Number(job.elapsedSeconds) || 0, elapsed);
}
