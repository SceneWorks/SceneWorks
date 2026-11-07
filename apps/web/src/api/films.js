import { modelsApi } from "./modelsResources.js";
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

export function getFilmPlanning(projectId, draftId, token, options = {}) {
  return apiFetch(`${base(projectId, draftId)}/planning`, token, options);
}

export function startFilmPlanning(projectId, draftId, maxRepairRounds, llmTimeoutSeconds, token) {
  return apiFetch(`${base(projectId, draftId)}/planning`, token, {
    method: "POST",
    body: JSON.stringify({ maxRepairRounds, llmTimeoutSeconds }),
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
  return modelsApi.download(modelId, token, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function preflightFilm(projectId, draftId, selectedShotIds, token, expectedDraftRevision, options = {}) {
  return apiFetch(`${base(projectId, draftId)}/preflight`, token, {
    ...options,
    method: "POST",
    body: JSON.stringify({ selectedShotIds, expectedDraftRevision }),
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

export function exportFilmRun(projectId, runId, token, options = {}) {
  return apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/film-runs/${encodeURIComponent(runId)}/export`, token, {
    ...options,
    method: "POST",
  });
}

export function filmRequest(projectId, draftId, token, options = {}, suffix = "") {
  const path = draftId == null
    ? `/api/v1/projects/${encodeURIComponent(projectId)}/films`
    : base(projectId, draftId);
  return apiFetch(`${path}${suffix}`, token, options);
}

export function listFilmRuns(projectId, token, options = {}) {
  return apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/film-runs`, token, options);
}

export function getFilmRun(projectId, runId, token, options = {}) {
  return apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/film-runs/${encodeURIComponent(runId)}`, token, options);
}

export function startFilmRun(projectId, runId, action, token, options = {}) {
  return apiFetch(`/api/v1/projects/${encodeURIComponent(projectId)}/film-runs/${encodeURIComponent(runId)}/${encodeURIComponent(action)}`, token, { ...options, method: "POST" });
}
