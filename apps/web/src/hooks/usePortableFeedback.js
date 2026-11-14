import { useEffect, useState } from "react";

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
