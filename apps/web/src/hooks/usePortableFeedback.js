import { useEffect, useRef, useState } from "react";
import { useAppStaticOptional } from "../context/AppContext.js";
import { MAX_VALIDATION_BATCH } from "../portableValidationState.js";

function submittedDraft(kind, payload, projectId) {
  // YuE2 takes its project from the URL and rejects unknown job-spec fields.
  return kind !== "yue2" && projectId ? { ...payload, projectId } : payload;
}

function useServerRetry(required) {
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!required) return;
    const retryNow = () => setRetry((value) => value + 1);
    const timer = setTimeout(retryNow, 10000);
    window.addEventListener("online", retryNow);
    return () => { clearTimeout(timer); window.removeEventListener("online", retryNow); };
  }, [required, retry]);
  return retry;
}

// Forms pass the exact payload they would submit. A stale response cannot unlock a newer draft.
export function usePortableFeedback(kind, payload, _modelEntry, enabled = true) {
  const app = useAppStaticOptional();
  const projectId = app?.activeProject?.id;
  const token = app?.token ?? "";
  let draft = null;
  if (enabled && payload) {
    try {
      draft = JSON.stringify(payload);
    } catch { /* Server validates the request. */ }
  }
  const [feedback, setFeedback] = useState(null);
  const scope = JSON.stringify([draft, projectId]);
  const retry = useServerRetry(feedback?.scope === scope && feedback.kind === kind
    && feedback.result.state === "server-required");
  useEffect(() => {
    if (!draft) return;
    let current = true;
    const controller = new AbortController();
    import("../portableValidation.js")
      .then(({ validatePortableRequest }) => validatePortableRequest({
        kind, payload: submittedDraft(kind, JSON.parse(draft), projectId), token, signal: controller.signal,
      }))
      .then((result) => {
        if (current) setFeedback({ scope, kind, result });
      })
      .catch(() => { if (current) setFeedback({ scope, kind, result: { state: "server-required" } }); });
    return () => { current = false; controller.abort(); };
  }, [draft, kind, projectId, scope, token, retry]);
  if (!draft) return null;
  return feedback?.scope === scope && feedback.kind === kind
    ? feedback.result : { state: "pending" };
}

// Inspect every authored item in bounded server batches; a new edit cancels the old response.
const BACKGROUND_BATCH_LIMIT = 256;
export function usePortableBatchFeedback(kind, revision, requests, _modelEntry, enabled = true) {
  const app = useAppStaticOptional();
  const projectId = app?.activeProject?.id;
  const token = app?.token ?? "";
  const requestsRef = useRef(requests);
  requestsRef.current = requests;
  const [feedback, setFeedback] = useState(null);
  const retry = useServerRetry(feedback?.kind === kind && feedback.revision === revision
    && feedback.projectId === projectId && feedback.result.state === "server-required");
  useEffect(() => {
    if (!enabled || !revision) return;
    let current = true;
    const controller = new AbortController();
    const inspect = async () => {
      const { validatePortableBatch } = await import("../portableValidation.js");
      let index = 0;
      let fallback = null;
      const iterator = requestsRef.current();
      while (true) {
        if (!current) return;
        const chunk = [];
        for (let slot = 0; slot < MAX_VALIDATION_BATCH; slot += 1) {
          const next = iterator.next();
          if (next.done) break;
          chunk.push(submittedDraft(kind, next.value, projectId));
        }
        if (!chunk.length) break;
        const results = await validatePortableBatch({ kind, requests: chunk, token, signal: controller.signal });
        if (!current) return;
        for (const result of results) {
          index += 1;
          if (result.state === "invalid") {
            setFeedback({ kind, revision, projectId, result, index });
            return;
          }
          if (result.state === "server-required") fallback = result;
        }
        if (fallback) break;
        if (index >= BACKGROUND_BATCH_LIMIT) {
          const more = !iterator.next().done;
          if (current) setFeedback({ kind, revision, projectId,
            result: more ? { state: "partial" } : { state: "valid" }, index });
          return;
        }
      }
      if (current) setFeedback({ kind, revision, projectId, result: fallback ?? { state: "valid" }, index });
    };
    void inspect().catch(() => {
      if (current) setFeedback({ kind, revision, projectId, result: { state: "server-required" }, index: 0 });
    });
    return () => { current = false; controller.abort(); };
  }, [enabled, kind, projectId, revision, token, retry]);
  if (!enabled || !revision) return null;
  return feedback?.kind === kind && feedback.revision === revision
    && feedback.projectId === projectId ? feedback : { result: { state: "pending" }, index: 0 };
}
