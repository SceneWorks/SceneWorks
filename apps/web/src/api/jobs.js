import { openEventStream } from "./client.js";
import { apiFetch, eventUrl } from "../api.js";

// Job producers (editor, batch, prompt tools) deliberately share endpoint values
// with their pure body builders. Keep route selection in this domain, including
// generic jobs and every specialized job surface supported by those builders.
export const JOB_PATHS = Object.freeze({
  generic: "/api/v1/jobs", video: "/api/v1/video/jobs", audio: "/api/v1/audio/jobs",
  image: "/api/v1/image/jobs", vqa: "/api/v1/image/vqa/jobs",
  interleave: "/api/v1/image/interleave/jobs", vector: "/api/v1/image/vectorize/jobs",
  vectorPrompt: "/api/v1/image/vectorize/prompt/jobs", depth: "/api/v1/image/depth/jobs",
  refine: "/api/v1/prompts/refine", compareFaces: "/api/v1/face-likeness/compare",
});
export function submitJobRequest(path, token, options) {
  return apiFetch(path, token, options);
}
export function jobActionRequest(jobId, action, token, options) {
  return apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/${encodeURIComponent(action)}`, token, options);
}
export function jobEventsUrl(ticket) {
  return eventUrl("/api/v1/jobs/events", ticket);
}

export function jobEventsSource(ticket) {
  return openEventStream(jobEventsUrl(ticket));
}
