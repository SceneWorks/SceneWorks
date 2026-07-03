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

// True when the page itself is in browser (HTML5) fullscreen. Only meaningful on the
// remote-browser path — desktop uses native window fullscreen, which leaves
// `document.fullscreenElement` null, so callers there track state themselves.
export function isPageFullscreen() {
  return typeof document !== "undefined" && !!document.fullscreenElement;
}

// Take the current viewer surface into (or out of) true OS-level fullscreen.
//
// Desktop (Tauri) drives the native window: `getCurrentWindow().setFullscreen(...)`.
// This is reliable in WKWebView, whereas element `requestFullscreen()` support in
// Tauri's macOS webview is not guaranteed — so we don't bet the primary platform on
// it. Requires the `core:window:allow-set-fullscreen` ACL grant in the desktop
// capability set.
//
// A remote browser (epic 4484) has no `window.__TAURI__`, so it falls back to the
// HTML5 Fullscreen API on the document element; the browser then owns Esc-to-exit and
// emits `fullscreenchange`, which the caller listens to for state sync.
export async function setViewerFullscreen(enabled) {
  if (isDesktop) {
    await window.__TAURI__.window.getCurrentWindow().setFullscreen(enabled);
    return;
  }
  if (typeof document === "undefined") {
    return;
  }
  if (enabled) {
    await document.documentElement?.requestFullscreen?.();
  } else if (document.fullscreenElement) {
    await document.exitFullscreen?.();
  }
}
