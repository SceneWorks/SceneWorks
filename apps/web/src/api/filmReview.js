import { apiFetch } from "../api.js";

function base(projectId, runId) {
  return `/api/v1/projects/${encodeURIComponent(projectId)}/film-runs/${encodeURIComponent(runId)}/review`;
}

export function getFilmReview(projectId, runId, token, options = {}) {
  return apiFetch(base(projectId, runId), token, options);
}

function post(projectId, runId, suffix, body, token, options = {}) {
  return apiFetch(`${base(projectId, runId)}${suffix}`, token, {
    ...options,
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function analyzeFilmTake(projectId, runId, shotIds, token, options = {}) {
  return post(projectId, runId, "", { shotIds }, token, options);
}

export function decideFilmTake(projectId, runId, shotId, decision, reason, token, options = {}) {
  return post(projectId, runId, "/decision", { shotId, decision, reason }, token, options);
}

export function swapFilmTake(projectId, runId, shotId, assetId, token, options = {}) {
  return post(projectId, runId, "/swap", { shotId, assetId }, token, options);
}

export function replaceFilmTake(projectId, runId, shotId, reason, token, options = {}) {
  return post(projectId, runId, "/replace", { shotId, reason }, token, options);
}

export function repairFilmTake(projectId, runId, shotId, reason, token, options = {}) {
  return post(projectId, runId, "/repair", { shotId, reason }, token, options);
}
