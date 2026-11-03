import { API_BASE_URL, fetchResource, withMediaTicket } from "../api.js";

// File keys retain their directory hierarchy; each component is independently
// encoded. A project/model/asset id is always a single route parameter.
export function projectFileUrl(projectId, filePath, { ticket = true, encodedPath = false, suffix = "" } = {}) {
  const path = encodedPath ? filePath : String(filePath).replaceAll("\\", "/").split("/").filter(Boolean).map(encodeURIComponent).join("/");
  const url = `${API_BASE_URL}/api/v1/projects/${encodeURIComponent(projectId)}/files/${path}${suffix}`;
  return ticket ? withMediaTicket(url) : url;
}
export function posePreviewUrl(jobId, skeletonFile) {
  return withMediaTicket(`${API_BASE_URL}/api/v1/poses/preview/${encodeURIComponent(jobId)}/${encodeURIComponent(skeletonFile)}`);
}
export function catalogThumbnailUrl(catalogId, recordId) {
  return withMediaTicket(`${API_BASE_URL}/api/v1/catalogs/${encodeURIComponent(catalogId)}/records/${encodeURIComponent(recordId)}/thumbnail`);
}
// Ticket-bearing media URLs may be data/blob URLs as well as API files. Preserve
// the browser Response/stream contract and caller-owned status/body consumption.
export const mediaResponse = (url, options) => fetchResource(url, options);
