import { useEffect, useRef, useState } from "react";

// Forms pass the exact payload they would submit. A stale async result never blocks a newer
// draft, and a cold import remains a server-owned check until the Rust module is ready.
export function usePortableFeedback(kind, payload, modelEntry, enabled = true) {
  let draft = null;
  if (enabled && payload) {
    try { draft = JSON.stringify(payload); } catch { /* Server validates the request. */ }
  }
  const [feedback, setFeedback] = useState(null);
  useEffect(() => {
    if (!draft) return;
    let current = true;
    import("../portableValidation.js")
      .then(({ validatePortableRequest }) => validatePortableRequest({
        kind, payload: JSON.parse(draft), modelEntry,
      }))
      .then((result) => {
        if (current) setFeedback({ draft, kind, modelEntry, result });
      })
      .catch(() => { if (current) setFeedback(null); });
    return () => { current = false; };
  }, [draft, kind, modelEntry]);
  return feedback?.draft === draft && feedback.kind === kind && feedback.modelEntry === modelEntry
    ? feedback.result : null;
}

// A prompt batch can expand to many distinct requests. Inspect every authored item without
// holding the input event: the iterator is consumed in small slices and abandoned on a new draft.
export function usePortableBatchFeedback(kind, revision, requests, modelEntry, enabled = true) {
  const requestsRef = useRef(requests);
  requestsRef.current = requests;
  const [feedback, setFeedback] = useState(null);
  useEffect(() => {
    if (!enabled || !revision) return;
    let current = true;
    const inspect = async () => {
      const { validatePortableRequest } = await import("../portableValidation.js");
      let index = 0;
      let fallback = null;
      for (const payload of requestsRef.current()) {
        if (!current) return;
        index += 1;
        const result = await validatePortableRequest({ kind, payload, modelEntry });
        if (!current) return;
        if (result.state === "invalid") {
          setFeedback({ revision, result, index });
          return;
        }
        if (result.state === "server-required") fallback = result;
        if (index % 8 === 0) await new Promise((resolve) => window.setTimeout(resolve, 0));
      }
      if (current) setFeedback({ revision, result: fallback ?? { state: "valid" }, index });
    };
    void inspect().catch(() => {
      if (current) setFeedback({ revision, result: { state: "server-required" }, index: 0 });
    });
    return () => { current = false; };
  }, [enabled, kind, modelEntry, revision]);
  return enabled && feedback?.revision === revision ? feedback : null;
}
