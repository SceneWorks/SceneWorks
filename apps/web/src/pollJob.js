import { apiFetch } from "./api.js";

// Resolve after `ms`, or reject with an AbortError if `signal` fires (or is already
// aborted). Shared by the poll-to-completion runners so a caller's AbortController
// cancels the in-flight 1s wait immediately rather than after the timer elapses.
export function abortableDelay(ms, signal) {
  if (signal?.aborted) {
    return Promise.reject(new DOMException("Aborted", "AbortError"));
  }
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

// Shared POST-a-job-then-poll-until-terminal runner (sc-8856). Five App.jsx callbacks
// (refinePrompt / magicPrompt / imageCaption / imageDescribe / compareFaceLikeness) all
// enqueue a worker job via a POST, then `while (Date.now() < deadline)` of
// `abortableDelay(1000)` + `GET /jobs/:id` until the job reaches a terminal status. This
// centralises that mechanism while keeping every per-caller variation explicit:
//   - createPath   : the enqueue endpoint (e.g. "/api/v1/prompts/refine").
//   - body         : the JSON body object to POST (stringified here).
//   - deadlineMs    : the caller's own poll deadline (refine=120s, the rest=180s).
//   - resolveResult : (job) => value — invoked once the job is "completed"; each caller
//                     extracts + validates its own result field here and throws its own
//                     "empty result" error, so the completion semantics stay per-caller.
//   - startError / failureError / timeoutError : the caller's own messages for a missing
//                     job id, a failed/canceled/interrupted terminal state, and a deadline
//                     timeout respectively.
// Terminal-status handling ("completed" vs failed/canceled/interrupted) and the 1s poll
// interval are identical across all five callers, so they live here; everything that
// varied stays a parameter. Throws on any non-success path; the `signal` cancels both the
// POST/GET fetches and the inter-poll delay.
export async function pollJobToCompletion({
  createPath,
  body,
  deadlineMs,
  resolveResult,
  signal,
  token,
  startError,
  failureError,
  timeoutError,
}) {
  const created = await apiFetch(createPath, token, {
    method: "POST",
    signal,
    body: JSON.stringify(body),
  });
  const jobId = created?.id;
  if (!jobId) {
    throw new Error(startError);
  }
  const deadline = Date.now() + deadlineMs;
  while (Date.now() < deadline) {
    await abortableDelay(1000, signal);
    const job = await apiFetch(`/api/v1/jobs/${jobId}`, token, { signal });
    if (job.status === "completed") {
      return resolveResult(job);
    }
    if (job.status === "failed" || job.status === "canceled" || job.status === "interrupted") {
      throw new Error(job.message || job.error || failureError);
    }
  }
  throw new Error(timeoutError);
}
