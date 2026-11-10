import { JOB_PATHS } from "./jobs.js";
import { apiFetch } from "../api.js";

// jobs route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const jobsApi = {
  list: (token, options = {}) =>
    apiFetch(JOB_PATHS.generic, token, options),
  clear: (token, options = {}) =>
    apiFetch(`/api/v1/jobs/clear`, token, options),
  cancelPending: (token, options = {}) =>
    apiFetch(`/api/v1/jobs/cancel-pending`, token, options),
  prioritize: (token, options = {}) =>
    apiFetch(`/api/v1/jobs/prioritize`, token, options),
  clearJob: (jobId, token, options = {}) =>
    apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/clear`, token, options),
  cancel: (jobId, token, options = {}) =>
    apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, token, options),
  item: (jobId, token, options = {}) =>
    apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`, token, options),
  // A page of a job's per-item results in order: `{ items, total, nextOffset }` (SW-7082).
  resultItems: (jobId, token, { offset = 0, limit } = {}, options = {}) =>
    apiFetch(
      `/api/v1/jobs/${encodeURIComponent(jobId)}/result-items?offset=${encodeURIComponent(offset)}${limit ? `&limit=${encodeURIComponent(limit)}` : ""}`,
      token,
      options,
    ),
  eventsTicket: (token, options = {}) =>
    apiFetch(`/api/v1/jobs/events/ticket`, token, options),
  duplicate: (jobId, token, options = {}) =>
    apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/duplicate`, token, options),
};
