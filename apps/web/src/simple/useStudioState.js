import { useEffect, useRef, useState } from "react";
import { useSimpleUi } from "./SimpleUiContext.js";

// Studio state that survives navigation AND a relaunch.
//
// The Simple shell renders exactly one screen at a time (`{screen === "image" ? … : null}`),
// so every studio UNMOUNTS when you leave it and its `useState` values are gone — walk to
// Assets and back and the model/prompt/resolution you chose have been re-seeded from the
// catalog defaults. `dismissedJobIds` already lives on the shell for precisely this reason;
// this is the same trick generalised, so a studio's own knobs get the same treatment.
//
// Two layers, because they solve two different problems:
//   values  — a plain Map on a shell-owned ref, NOT React state: it is written on every
//             keystroke and must not re-render the shell (and therefore every screen). This
//             is what survives NAVIGATION.
//   persist — the shell's debounced write-through to `simpleStudioStore`, which is what
//             survives a RELAUNCH (localStorage cache + the durable ui-preferences copy).
//
// The mounted studio still holds the live value in its own `useState`; both layers are only
// carbon copies that outlive it.

export function createStudioStateStore() {
  return {
    values: new Map(),
    // Replaced by the shell once it can persist. A no-op until then, so a studio that renders
    // before the shell has wired durability simply doesn't write — it never throws and never
    // pushes a half-seeded snapshot at the store.
    persist: () => {},
  };
}

/** The `{ [scope]: { [key]: value } }` snapshot for a store's Map — the shape the store persists. */
export function snapshotFromStore(store) {
  const snapshot = {};
  for (const [id, value] of store.values) {
    const separator = id.indexOf(":");
    if (separator <= 0) {
      continue;
    }
    const scope = id.slice(0, separator);
    const key = id.slice(separator + 1);
    snapshot[scope] = { ...(snapshot[scope] ?? {}), [key]: value };
  }
  return snapshot;
}

/** Load a persisted `{ [scope]: { [key]: value } }` snapshot into a store's Map, replacing it. */
export function loadSnapshotIntoStore(store, snapshot) {
  store.values.clear();
  if (!snapshot || typeof snapshot !== "object") {
    return;
  }
  for (const [scope, fields] of Object.entries(snapshot)) {
    if (!fields || typeof fields !== "object") {
      continue;
    }
    for (const [key, value] of Object.entries(fields)) {
      store.values.set(`${scope}:${key}`, value);
    }
  }
}

/**
 * `useState`, but the value is written through to the shell's store and read back from it
 * when the studio remounts.
 *
 * @param {string} scope - the owning studio ("image" | "video" | "audio"), so two studios
 *   can each keep their own `model` without colliding.
 * @param {string} key - the field name.
 * @param {*} initial - the value used when this scope+key has no stored value.
 */
export function useStudioState(scope, key, initial) {
  const { studioState } = useSimpleUi();
  const id = `${scope}:${key}`;
  // Read once, at mount. Re-reading on every render would fight the local state below — which
  // is also why the shell REMOUNTS the studios (via a key) when a restored snapshot lands,
  // rather than trying to push new values into already-mounted state.
  const [value, setValue] = useState(() =>
    studioState.values.has(id) ? studioState.values.get(id) : initial,
  );

  // Write-through in an effect rather than inside the setter: a state updater may be invoked
  // more than once for a single update (StrictMode, concurrent re-render), and a Map write is
  // a side effect that does not belong in one.
  const idRef = useRef(id);
  idRef.current = id;
  useEffect(() => {
    studioState.values.set(idRef.current, value);
    studioState.persist();
  }, [studioState, value]);

  // `setValue` is React's own `useState` setter, so it is referentially stable and safe to
  // list in a dependency array — which callers must do, because exhaustive-deps only grants
  // the implicit-stability exemption to a setter destructured straight off `useState`.
  return [value, setValue];
}
