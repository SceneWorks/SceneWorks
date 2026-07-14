import { createContext, useContext } from "react";

// Selective keep-alive (sc-11959): whether the screen reading this context is the
// currently-active view. Under the keep-alive shell a creative screen stays mounted
// after its first visit and is merely hidden when another view is active, so it can no
// longer rely on unmount/remount to learn it was backgrounded. This context surfaces
// that signal so a screen can pause expensive work while hidden (consumed by a later
// story, S2) or drop an editor leave-guard when it isn't the foreground view.
//
// Defaults to `true`: any screen rendered OUTSIDE a keep-alive pane — the OUT screens
// (still conditionally mounted) and unit tests that render a screen directly under a
// bare provider — reads as "active", preserving today's behavior with no wiring.
export const ScreenActiveContext = createContext(true);

// Read whether the current screen is the active view. Returns `true` unless a
// KeepAlivePane above it says otherwise.
export function useScreenActive() {
  return useContext(ScreenActiveContext);
}
