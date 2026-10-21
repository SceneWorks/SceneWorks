import { useEffect, useState } from "react";

import { fetchTrainingDataWalk, isAbortError } from "../api.js";

// The Iris data-walk preflight (sc-25685): re-asks the API whenever the selected Iris target, the
// saved dataset's item count, or the batch size changes, and hands back the
// `{ applies, covered, unusedItems, warning, ... }` result. Non-blocking by design — a failed
// preflight just shows no warning; the API still validates the run itself.
export function useIrisDataWalk({ enabled, targetId, items, batchSize, token }) {
  const [result, setResult] = useState(null);
  const itemCount = Number(items);
  const batch = Number(batchSize);
  const ready = Boolean(enabled && targetId && Number.isInteger(itemCount) && itemCount > 0 && Number.isInteger(batch) && batch > 0);
  useEffect(() => {
    if (!ready) {
      setResult(null);
      return undefined;
    }
    const controller = new AbortController();
    fetchTrainingDataWalk({ targetId, items: itemCount, batchSize: batch }, { token, signal: controller.signal })
      .then((walk) => {
        if (!controller.signal.aborted) setResult(walk ?? null);
      })
      .catch((err) => {
        if (!isAbortError(err) && !controller.signal.aborted) setResult(null);
      });
    return () => controller.abort();
  }, [ready, targetId, itemCount, batch, token]);
  return ready ? result : null;
}
