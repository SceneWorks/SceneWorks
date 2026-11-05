import { useLayoutEffect, useMemo, useRef } from "react";

// Event handlers retain stable identities while invoking the current render's logic.
export function useStableActions(actions) {
  const latest = useRef(actions);
  useLayoutEffect(() => { latest.current = actions; });
  return useMemo(() => Object.fromEntries(Object.keys(latest.current).map((key) =>
    [key, (...args) => latest.current[key](...args)])), []);
}

// Per-item job/cache selections are freshly filtered arrays of stable records.
export function equalItemProps(previous, next) {
  const keys = Object.keys(previous);
  return keys.length === Object.keys(next).length && keys.every((key) => {
    const a = previous[key];
    const b = next[key];
    return Object.is(a, b) || (Array.isArray(a) && Array.isArray(b) &&
      a.length === b.length && a.every((item, index) => Object.is(item, b[index])));
  });
}
