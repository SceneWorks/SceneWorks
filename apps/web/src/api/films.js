import { apiFetch } from "../api.js";

function base(projectId, draftId) {
  return `/api/v1/projects/${encodeURIComponent(projectId)}/films/${encodeURIComponent(draftId)}`;
}

export function parseFilmScript(projectId, draftId, script, token) {
  return apiFetch(`${base(projectId, draftId)}/brief/parse`, token, {
    method: "POST",
    body: JSON.stringify({ script }),
  });
}

export function getFilmPlannerAvailability(projectId, draftId, token) {
  return apiFetch(`${base(projectId, draftId)}/planners`, token);
}

export function getFilmPlanning(projectId, draftId, token) {
  return apiFetch(`${base(projectId, draftId)}/planning`, token);
}

export function startFilmPlanning(projectId, draftId, token) {
  return apiFetch(`${base(projectId, draftId)}/planning`, token, {
    method: "POST",
    body: JSON.stringify({ maxRepairRounds: 2 }),
  });
}

export function cancelFilmPlanning(projectId, draftId, token) {
  return apiFetch(`${base(projectId, draftId)}/planning/cancel`, token, { method: "POST" });
}

export function applyFilmPlanning(projectId, draftId, operationId, token) {
  return apiFetch(`${base(projectId, draftId)}/planning/apply`, token, {
    method: "POST",
    body: JSON.stringify({ operationId }),
  });
}

export function installFilmPlanner(modelId, token) {
  return apiFetch(`/api/v1/models/${encodeURIComponent(modelId)}/download`, token, {
    method: "POST",
    body: JSON.stringify({}),
  });
}
