import { useEffect, useRef, useState } from "react";
import { useSimpleUi } from "./SimpleUiContext.js";

// Studio state that survives navigation.
//
// The Simple shell renders exactly one screen at a time (`{screen === "image" ? … : null}`),
// so every studio UNMOUNTS when you leave it and its `useState` values are gone — walk to
// Assets and back and the model/prompt/resolution you chose have been re-seeded from the
// catalog defaults. `dismissedJobIds` already lives on the shell for precisely this reason;
// this is the same trick generalised, so a studio's own knobs get the same treatment.
//
// The store is a plain Map on a shell-owned ref, NOT React state: writing to it must not
// re-render the shell (and therefore every screen) on each keystroke. The mounted studio
// already holds the live value in its own `useState` — the store is only the carbon copy
// that outlives it.
//
// Session-scoped on purpose. This restores where you were while the app is open; it is not
// a preferences store, so a relaunch still starts from the catalog defaults.

export function createStudioStateStore() {
  return new Map();
}

/**
 * `useState`, but the value is written through to the shell's store and read back from it
 * when the studio remounts.
 *
 * @param {string} scope - the owning studio ("image" | "video" | "audio"), so two studios
 *   can each keep their own `model` without colliding.
 * @param {string} key - the field name.
 * @param {*} initial - the value used the FIRST time this scope+key is seen in a session.
 */
export function useStudioState(scope, key, initial) {
  const { studioState } = useSimpleUi();
  const id = `${scope}:${key}`;
  // Read once, at mount. Re-reading on every render would fight the local state below.
  const [value, setValue] = useState(() => (studioState.has(id) ? studioState.get(id) : initial));

  // Write-through in an effect rather than inside the setter: a state updater may be invoked
  // more than once for a single update (StrictMode, concurrent re-render), and a Map write is
  // a side effect that does not belong in one.
  const idRef = useRef(id);
  idRef.current = id;
  useEffect(() => {
    studioState.set(idRef.current, value);
  }, [studioState, value]);

  // `setValue` is React's own `useState` setter, so it is referentially stable and safe to
  // list in a dependency array — which callers must do, because exhaustive-deps only grants
  // the implicit-stability exemption to a setter destructured straight off `useState`.
  return [value, setValue];
}
