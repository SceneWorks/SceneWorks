import { terminalStatuses } from "./jobTypes.js";

export function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) {
    return "0s";
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return minutes > 0 ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

export function percent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
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
