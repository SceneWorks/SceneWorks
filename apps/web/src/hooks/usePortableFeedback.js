import { useEffect, useRef, useState } from "react";
import { useAppStaticOptional } from "../context/AppContext.js";
import { MAX_VALIDATION_BATCH } from "../portableValidationState.js";

function submittedDraft(payload, projectId) {
  return projectId ? { ...payload, projectId } : payload;
}

// Forms pass the exact payload they would submit. A stale response cannot unlock a newer draft.
export function usePortableFeedback(kind, payload, _modelEntry, enabled = true) {
  const app = useAppStaticOptional();
  const projectId = app?.activeProject?.id;
  const token = app?.token ?? "";
  let draft = null;
  if (enabled && payload) {
    try {
      draft = JSON.stringify(submittedDraft(payload, projectId));
    } catch { /* Server validates the request. */ }
  }
  const [feedback, setFeedback] = useState(null);
  useEffect(() => {
    if (!draft) return;
    let current = true;
    const controller = new AbortController();
    import("../portableValidation.js")
      .then(({ validatePortableRequest }) => validatePortableRequest({
        kind, payload: JSON.parse(draft), token, signal: controller.signal,
      }))
      .then((result) => {
        if (current) setFeedback({ draft, kind, result });
      })
      .catch(() => { if (current) setFeedback({ draft, kind, result: { state: "server-required" } }); });
    return () => { current = false; controller.abort(); };
  }, [draft, kind, token]);
  if (!draft) return null;
  return feedback?.draft === draft && feedback.kind === kind
    ? feedback.result : { state: "pending" };
}

// Inspect every authored item in bounded server batches; a new edit cancels the old response.
export function usePortableBatchFeedback(kind, revision, requests, _modelEntry, enabled = true) {
  const app = useAppStaticOptional();
  const projectId = app?.activeProject?.id;
  const token = app?.token ?? "";
  const requestsRef = useRef(requests);
  requestsRef.current = requests;
  const [feedback, setFeedback] = useState(null);
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
          chunk.push(submittedDraft(next.value, projectId));
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
      }
      if (current) setFeedback({ kind, revision, projectId, result: fallback ?? { state: "valid" }, index });
    };
    void inspect().catch(() => {
      if (current) setFeedback({ kind, revision, projectId, result: { state: "server-required" }, index: 0 });
    });
    return () => { current = false; controller.abort(); };
  }, [enabled, kind, projectId, revision, token]);
  if (!enabled || !revision) return null;
  return feedback?.kind === kind && feedback.revision === revision
    && feedback.projectId === projectId ? feedback : { result: { state: "pending" }, index: 0 };
}
