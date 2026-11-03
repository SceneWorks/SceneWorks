import { useInsertionEffect, useMemo, useRef } from "react";

// Fixed handler keys, current render closures. A memoized consumer may retain the
// wrapper without retaining an old dataset, project, token or draft snapshot.
export function useLatestHandlers(handlers) {
  const latest = useRef(handlers);
  // Commit before descendant layout effects, without exposing a suspended or
  // abandoned render through callbacks that are still visible in the current UI.
  useInsertionEffect(() => { latest.current = handlers; }, [handlers]);
  return useMemo(() => Object.fromEntries(
    Object.keys(latest.current).map((key) => [key, (...args) => latest.current[key]?.(...args)]),
  ), []);
}
