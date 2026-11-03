import { useCallback, useEffect, useRef, useState } from "react";
import { fetchModelCache } from "../modelCache.js";
import { useCacheConvergence } from "./useCacheConvergence.js";

// Both consumers share one read and bounded convergence; the manager suppresses
// polling while disabled, and treats a snapshot error as a global control error.
export function useModelCacheStatus(token, { pollWhenDisabled = true, reportSnapshotError = false } = {}) {
  const [cache, setCache] = useState(null);
  const [error, setError] = useState("");
  const sequence = useRef(0);
  const refresh = useCallback(async () => {
    const request = ++sequence.current;
    try {
      const snapshot = await fetchModelCache(token);
      if (request === sequence.current) {
        setCache(snapshot);
        setError(reportSnapshotError && snapshot?.error ? String(snapshot.error) : "");
      }
      return snapshot;
    } catch (failure) {
      if (request === sequence.current) {
        setCache(null);
        setError(String(failure?.message ?? failure));
      }
      return null;
    }
  }, [token, reportSnapshotError]);
  useEffect(() => {
    setCache(null);
    setError("");
    refresh();
    return () => { sequence.current += 1; };
  }, [refresh]);
  const { stalled } = useCacheConvergence(pollWhenDisabled || cache?.policy?.enabled === true ? cache : null, refresh);
  return { cache, error, refresh, stalled };
}
