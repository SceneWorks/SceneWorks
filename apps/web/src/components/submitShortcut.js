// The studio prompt boxes submit on Cmd+Enter or Ctrl+Enter (both are accepted); the Generate /
// Render clip button names the one this platform's users reach for.
const IS_MAC =
  typeof navigator !== "undefined" && /Mac|iP(hone|ad|od)/.test(navigator.platform || navigator.userAgent || "");

export const SUBMIT_SHORTCUT_LABEL = IS_MAC ? "⌘ ↵" : "Ctrl ↵";
export const SUBMIT_SHORTCUT_ARIA = IS_MAC ? "Meta+Enter" : "Control+Enter";
