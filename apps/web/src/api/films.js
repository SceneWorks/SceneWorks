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

export function getFilmPlannerAvailability(projectId, draftId, token, options = {}) {
  return apiFetch(`${base(projectId, draftId)}/planners`, token, options);
}

export function getFilmPlanning(projectId, draftId, token) {
  return apiFetch(`${base(projectId, draftId)}/planning`, token);
}

export function startFilmPlanning(projectId, draftId, maxRepairRounds, token) {
  return apiFetch(`${base(projectId, draftId)}/planning`, token, {
    method: "POST",
    body: JSON.stringify({ maxRepairRounds }),
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

export function preflightFilm(projectId, draftId, selectedShotIds, token) {
  return apiFetch(`${base(projectId, draftId)}/preflight`, token, {
    method: "POST",
    body: JSON.stringify({ selectedShotIds }),
  });
}

export function getFilmRenderOptions(projectId, draftId, token, options = {}) {
  return apiFetch(`${base(projectId, draftId)}/render-options`, token, options);
}

export function previewFilmRenderOptions(projectId, draftId, draft, token, options = {}) {
  return apiFetch(`${base(projectId, draftId)}/render-options`, token, {
    ...options,
    method: "POST",
    body: JSON.stringify({
      draftRevision: draft.revision,
      productionPlan: draft.productionPlan,
      referencePack: draft.referencePack,
      renderRegime: draft.renderRegime ?? "custom",
    }),
  });
}

export function addFilmSound(projectId, draftId, payload, token) {
  return apiFetch(`${base(projectId, draftId)}/sound`, token, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function exportFilmRun(projectId, runId, token) {
  return apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/film-runs/${encodeURIComponent(runId)}/export`, token, {
    method: "POST",
  });
}
