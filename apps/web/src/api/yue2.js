import { apiFetch } from "../api.js";

// YuE2 (epic 22988) client seam: the song-job route, the server-side licence acknowledgment and
// the score-version / comparison records. Every call goes through `apiFetch`, so a refusal arrives
// as an `ApiError` carrying the server's typed `code` and `context` (for example `component_blocked`
// with the blocked components' reasons, or `yue2_invariant_violation` with the invariant report).
// Nothing here swallows an error — the lab renders each one.

const enc = encodeURIComponent;

function projectBase(projectId) {
  return `/api/v1/projects/${enc(projectId)}/yue2`;
}

// POST /api/v1/projects/:id/yue2/jobs → { jobs, batchId }.
export function submitYue2Jobs(projectId, body, token) {
  return apiFetch(`${projectBase(projectId)}/jobs`, token, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// GET/PUT/DELETE /api/v1/models/:id/license-acknowledgment → { modelId, required, termsSha256,
// acknowledged, acknowledgment }. The acknowledgment is stored server-side for the entry's CURRENT
// terms, so it survives a relaunch and lapses when the terms change.
export function getLicenseAcknowledgment(modelId, token) {
  return apiFetch(`/api/v1/models/${enc(modelId)}/license-acknowledgment`, token);
}

export function putLicenseAcknowledgment(modelId, token) {
  return apiFetch(`/api/v1/models/${enc(modelId)}/license-acknowledgment`, token, { method: "PUT" });
}

export function deleteLicenseAcknowledgment(modelId, token) {
  return apiFetch(`/api/v1/models/${enc(modelId)}/license-acknowledgment`, token, { method: "DELETE" });
}

// Stateless inspection of any ABC text (exact events, or a typed 422 for unsupported notation).
export function inspectYue2Score(abc, token) {
  return apiFetch("/api/v1/yue2/score/inspect", token, {
    method: "POST",
    body: JSON.stringify({ abc }),
  });
}

export function listYue2ScoreVersions(projectId, token) {
  return apiFetch(`${projectBase(projectId)}/score-versions`, token);
}

export function getYue2ScoreVersion(projectId, versionId, token) {
  return apiFetch(`${projectBase(projectId)}/score-versions/${enc(versionId)}`, token);
}

export function inspectYue2ScoreVersion(projectId, versionId, token) {
  return apiFetch(`${projectBase(projectId)}/score-versions/${enc(versionId)}/inspection`, token);
}

// A root version: { abc, request: { style, lyrics, cot, seed?, cfgScale? }, origin, provenance }.
export function createYue2ScoreVersion(projectId, body, token) {
  return apiFetch(`${projectBase(projectId)}/score-versions`, token, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// { operation, brief, provenance, dryRun } → { dryRun, version, renderNotice }.
export function editYue2ScoreVersion(projectId, versionId, body, token) {
  return apiFetch(`${projectBase(projectId)}/score-versions/${enc(versionId)}/edits`, token, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function listYue2Comparisons(projectId, token) {
  return apiFetch(`${projectBase(projectId)}/comparisons`, token);
}

// { versionA, versionB, renderA?, renderB?, notes?, provenance } → the comparison record.
export function createYue2Comparison(projectId, body, token) {
  return apiFetch(`${projectBase(projectId)}/comparisons`, token, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
