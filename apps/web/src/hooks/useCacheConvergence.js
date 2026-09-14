import { useEffect, useState } from "react";
import {
  CACHE_CONVERGENCE_POLL_MS,
  MAX_CACHE_CONVERGENCE_POLLS,
  hasTransitionalEntries,
} from "../modelCache.js";

// The bounded convergence refresh for the resolved-model hot cache (sc-19711, epic 19703).
//
// Both cache surfaces — the Settings storage card and the Model Manager's per-model local-copy
// blocks — need the same behaviour, so it lives here ONCE rather than as two copies that can drift
// into disagreeing about when the app stops polling.
//
// Three properties are the whole point:
//
// 1. **It runs only while the store is actually working.** The status read is one journal listing,
//    proportional to the number of cached bundles; an unconditional timer over it would put back
//    exactly the per-row read cost sc-19708 took off these screens. A settled cache is read once.
// 2. **It is BOUNDED.** An entry that never converges — a worker that died mid-copy — must not
//    leave a screen re-reading for the life of the session. After `MAX_CACHE_CONVERGENCE_POLLS`
//    consecutive refreshes with something still in flight, it gives up and reports `stalled`, so
//    the caller can say so instead of leaving a "checking…" line that stopped meaning anything.
// 3. **It resets.** The spend counter clears the moment every entry is terminal, so a promotion
//    that starts later gets a fresh budget rather than inheriting an exhausted one.
//
// `refresh` must be referentially stable (a `useCallback`); it is a dependency of the timer effect.
export function useCacheConvergence(status, refresh) {
  const [polls, setPolls] = useState(0);
  const converging = hasTransitionalEntries(status);

  useEffect(() => {
    if (!converging) {
      setPolls(0);
      return undefined;
    }
    if (polls >= MAX_CACHE_CONVERGENCE_POLLS) return undefined;
    const timer = setTimeout(() => {
      setPolls((spent) => spent + 1);
      refresh();
    }, CACHE_CONVERGENCE_POLL_MS);
    return () => clearTimeout(timer);
  }, [converging, polls, refresh]);

  return {
    // Something is still in flight, whether or not this hook is still watching it.
    converging,
    // Still in flight, but nothing is re-reading any more.
    stalled: converging && polls >= MAX_CACHE_CONVERGENCE_POLLS,
  };
}
