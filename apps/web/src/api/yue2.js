import { API_BASE_URL, ApiError, apiFetch } from "../api.js";

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

// ---- cover from a recording (sc-23002) ------------------------------------------------------

// POST /api/v1/models/:id/conditional-components/:purpose/download
// { licenseAcknowledged, requestedGpu? } → { purpose, jobs, components: [{ componentId, repo,
// revision, status: "queued" | "installed", jobId? }] }. Refusals: 403 `license_acknowledgment_required`,
// 403 `component_blocked` (context.blocked[]), 404 `conditional_components_not_declared`.
export function downloadYue2ConditionalComponents(modelId, purpose, body, token) {
  return apiFetch(`/api/v1/models/${enc(modelId)}/conditional-components/${enc(purpose)}/download`, token, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// GET → { items: [TranscriptionRecord], unreadable, renderNotice }.
export function listYue2Transcriptions(projectId, token) {
  return apiFetch(`${projectBase(projectId)}/transcriptions`, token);
}

// GET → { transcription: TranscriptionRecord, scores: { full: abc | null, melody: abc | null } }.
export function getYue2Transcription(projectId, transcriptionId, token) {
  return apiFetch(`${projectBase(projectId)}/transcriptions/${enc(transcriptionId)}`, token);
}

// One export of a transcription's review artifact (MIDI, LAB, ABC, JSON), by its manifest path.
export function yue2TranscriptionFileUrl(projectId, transcriptionId, path) {
  const file = String(path ?? "")
    .split("/")
    .map(enc)
    .join("/");
  return `${API_BASE_URL}${projectBase(projectId)}/transcriptions/${enc(transcriptionId)}/files/${file}`;
}

// The export's bytes, fetched with the session token (the files route is not a media-ticket route,
// so a bare link would be refused in remote-auth mode). A refusal is an `ApiError` carrying the
// server's typed code — never an empty file.
export async function fetchYue2TranscriptionFile(projectId, transcriptionId, path, token) {
  const headers = new Headers();
  if (token) {
    headers.set("X-SceneWorks-Token", token);
  }
  const response = await fetch(yue2TranscriptionFileUrl(projectId, transcriptionId, path), { headers });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = payload?.detail;
    throw new ApiError(typeof detail === "string" ? detail : `Request failed with ${response.status}`, {
      status: response.status,
      detail,
      code: payload?.code,
      context: payload?.context,
    });
  }
  return response.blob();
}
