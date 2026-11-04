import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { fetchModelCache } from "../modelCache.js";
import { useCacheConvergence } from "./useCacheConvergence.js";

// Both consumers share one read and bounded convergence; the manager suppresses
// polling while disabled, and treats a snapshot error as a global control error.
export function useModelCacheStatus(token, { pollWhenDisabled = true, reportSnapshotError = false } = {}) {
  const [cache, setCache] = useState(null);
  const [error, setError] = useState("");
  const sequence = useRef(0);
  const owner = useMemo(() => ({ token, reportSnapshotError }), [token, reportSnapshotError]);
  const committedOwner = useRef(null);
  useLayoutEffect(() => {
    committedOwner.current = owner;
    return () => {
      committedOwner.current = null;
      sequence.current += 1;
    };
  }, [owner]);
  const refresh = useCallback(async () => {
    if (committedOwner.current !== owner) return null;
    const request = ++sequence.current;
    try {
      const snapshot = await fetchModelCache(owner.token);
      if (committedOwner.current === owner && request === sequence.current) {
        setCache(snapshot);
        setError(owner.reportSnapshotError && snapshot?.error ? String(snapshot.error) : "");
      }
      return snapshot;
    } catch (failure) {
      if (committedOwner.current === owner && request === sequence.current) {
        setCache(null);
        setError(String(failure?.message ?? failure));
      }
      return null;
    }
  }, [owner]);
  useEffect(() => {
    setCache(null);
    setError("");
    refresh();
  }, [refresh]);
  const { stalled } = useCacheConvergence(pollWhenDisabled || cache?.policy?.enabled === true ? cache : null, refresh);
  return { cache, error, refresh, stalled };
}
