import { API_BASE_URL, withMediaTicket } from "../api.js";

// Subject masks for training datasets (epic 2123, sc-2126). The upload limits mirror the API's
// (`crates/sceneworks-core/src/training_subject_masks.rs`): SUBJECT_MASK_MAX_UPLOAD_BYTES and the
// PNG/JPEG/WebP type set. The server re-validates and answers a field-level error on `file`; the
// editor checks first so an oversized or wrong-type file never leaves the browser.
export const SUBJECT_MASK_MAX_UPLOAD_BYTES = 32 * 1024 * 1024;
export const SUBJECT_MASK_ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp"];
export const SUBJECT_MASK_ACCEPT = SUBJECT_MASK_ACCEPTED_TYPES.join(",");

// "" when `file` may be uploaded as a subject mask, else the field error to show. A file with no
// reported type (some platforms) is left to the server's content sniff rather than refused here.
export function subjectMaskFileError(file) {
  if (!file) {
    return "Choose a mask image to upload.";
  }
  if (file.type && !SUBJECT_MASK_ACCEPTED_TYPES.includes(file.type)) {
    return "Subject mask must be a PNG, JPEG, or WebP image.";
  }
  if (file.size > SUBJECT_MASK_MAX_UPLOAD_BYTES) {
    return `Subject mask is larger than ${SUBJECT_MASK_MAX_UPLOAD_BYTES / (1024 * 1024)} MiB.`;
  }
  return "";
}

// Map item id → that item's mask status from a /subject-masks report.
export function subjectMaskStatusByItemId(report) {
  return new Map((report?.items ?? []).map((item) => [item.itemId, item]));
}

// Displayable URL of a mask. The mask path is fixed per image (content-hash keyed), so the stored
// PNG's content-derived `revision` is appended: a replaced mask is refetched instead of served from
// the browser cache, even when two replacements land within the same second.
export function subjectMaskUrl(projectId, status) {
  if (!projectId || !status?.maskPath) {
    return "";
  }
  const path = String(status.maskPath)
    .split("/")
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  const version = status.revision ? `?v=${encodeURIComponent(status.revision)}` : "";
  return withMediaTicket(`${API_BASE_URL}/api/v1/projects/${projectId}/files/${path}${version}`);
}
