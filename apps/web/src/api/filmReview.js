import { apiFetch } from "../api.js";

function base(projectId, runId) {
  return `/api/v1/projects/${encodeURIComponent(projectId)}/film-runs/${encodeURIComponent(runId)}/review`;
}

export function getFilmReview(projectId, runId, token) {
  return apiFetch(base(projectId, runId), token);
}

function post(projectId, runId, suffix, body, token) {
  return apiFetch(`${base(projectId, runId)}${suffix}`, token, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function analyzeFilmTake(projectId, runId, shotIds, token) {
  return post(projectId, runId, "", { shotIds }, token);
}

export function decideFilmTake(projectId, runId, shotId, decision, reason, token) {
  return post(projectId, runId, "/decision", { shotId, decision, reason }, token);
}

export function swapFilmTake(projectId, runId, shotId, assetId, token) {
  return post(projectId, runId, "/swap", { shotId, assetId }, token);
}

export function replaceFilmTake(projectId, runId, shotId, reason, token) {
  return post(projectId, runId, "/replace", { shotId, reason }, token);
}

export function repairFilmTake(projectId, runId, shotId, reason, token) {
  return post(projectId, runId, "/repair", { shotId, reason }, token);
}
