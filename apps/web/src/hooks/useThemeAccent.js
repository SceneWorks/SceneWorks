import { useCallback, useEffect, useState } from "react";
import { putUiPreferences } from "../uiPreferences.js";
import { readStoredAccent, readStoredTheme } from "../appHelpers.js";

// F-189: extracted from App along its existing concern boundary.
export function useThemeAccent() {
  const [theme, setTheme] = useState(readStoredTheme);

  // Apply a theme and persist it through the API. localStorage gives an instant
  // initial paint, but on the desktop shell the UI runs at the API's per-launch
  // http://127.0.0.1:<port> origin, where both localStorage and Tauri IPC are
  // unreliable across launches — so the durable copy lives server-side.
  // `putUiPreferences` carries the access token (sc-15136): the PUT is gated even
  // though the launch-time GET below is public, so a credential-less write 401s on
  // every remote-auth deployment and the theme would never reach disk.
  // Stable across renders (sc-10244): it flows through the static app context to the
  // Image Editor top-bar theme toggle, so a fresh identity each render would bust the
  // static-context memo every tick.
  const changeTheme = useCallback((next) => {
    setTheme(next);
    putUiPreferences({ theme: next }).catch(() => {});
  }, []);

  const [accent, setAccent] = useState(readStoredAccent);

  // Same persistence contract as theme: instant localStorage cache + durable
  // server copy. The PUT sends only the changed field, so the endpoint must
  // MERGE partial updates (theme writes already rely on this).
  const changeAccent = useCallback((next) => {
    setAccent(next);
    putUiPreferences({ accent: next }).catch(() => {});
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    document.documentElement.setAttribute("data-theme", theme);
    try {
      window.localStorage.setItem("sceneworks-theme", theme);
    } catch {
      // ignore (private mode etc.)
    }
  }, [theme]);

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    document.documentElement.setAttribute("data-accent", accent);
    try {
      window.localStorage.setItem("sceneworks-accent", accent);
    } catch {
      // ignore (private mode etc.)
    }
  }, [accent]);
  return { theme, setTheme, changeTheme, accent, setAccent, changeAccent };
}
