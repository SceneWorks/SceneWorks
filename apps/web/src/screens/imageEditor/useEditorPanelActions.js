import { useInsertionEffect, useMemo, useRef } from "react";

// Event callbacks only. Render helpers are ordinary props/imports so a render
// always derives its output from that render's data, including suspended work.
export function useEditorPanelActions(actions) {
  const committed = useRef(actions);
  useInsertionEffect(() => { committed.current = actions; }, [actions]);
  const keys = Object.keys(actions).filter((key) => typeof actions[key] === "function");
  const signature = JSON.stringify(keys);
  return useMemo(() => Object.fromEntries(JSON.parse(signature).map((key) => [key,
    (...args) => committed.current[key]?.(...args),
  ])), [signature]); // Keys alone determine these event wrappers; closures read committed owners.
}
