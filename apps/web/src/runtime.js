// Single source of truth for "are we running inside the Tauri desktop shell?"
// (epic 4484, story 6).
//
// Previously credentials.js, App.jsx, LogsScreen.jsx, ModelManagerScreen.jsx, and
// SettingsScreen.jsx each re-derived this from `window.__TAURI__` with subtly
// different expressions (one used `Boolean(...)`, the rest `!!...`). This unifies them
// so desktop-vs-remote-browser gating is consistent everywhere.
//
// A remote browser reaching the desktop host over the LAN (the whole point of
// epic 4484) has no `window.__TAURI__`, so `isDesktop` is false there and every
// desktop-only affordance falls back to its REST/web code path.
export const isDesktop = typeof window !== "undefined" && !!window.__TAURI__;

// Invoke a Tauri command. Only valid when `isDesktop` is true — callers MUST gate on
// it, since `window.__TAURI__` is undefined in a remote browser.
export function tauriInvoke(command, args) {
  return window.__TAURI__.core.invoke(command, args);
}
