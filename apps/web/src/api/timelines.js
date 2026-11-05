import { apiFetch } from "../api.js";

export function timelineRequest(projectId, timelineId, token, options = {}, suffix = "") {
  const base = `/api/v1/projects/${encodeURIComponent(projectId)}/timelines`;
  const path = timelineId == null ? base : `${base}/${encodeURIComponent(timelineId)}`;
  return apiFetch(`${path}${suffix}`, token, options);
}
