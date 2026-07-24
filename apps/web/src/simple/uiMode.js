import { isPhoneWidth } from "./breakpoint.js";

// Simple ⇄ Advanced UI mode (design handoff "Simple UI for creative studios").
//
// The Simple UI is an ALTERNATIVE shell, not a replacement: the full workspace is
// untouched and the sidebar switch flips between them. Three inputs decide which
// shell renders, in this precedence:
//
//   1. phone viewport  — always Simple (the full workspace is unusable at that width;
//                        the design's "Phones always use it"). Not overridable, so the
//                        sidebar switch is rendered disabled there.
//   2. session override — the user clicked Simple/Advanced in the sidebar this session.
//   3. stored default   — the "Use Simple UI by default" toggle in Settings.
//
// The stored default is OFF by default: an existing install keeps the workspace it
// already has, and opting into Simple is a deliberate choice (either the switch or the
// Settings toggle). Persistence mirrors theme/accent — a durable server copy in
// ui-preferences.json plus a localStorage instant-paint cache — because on the desktop
// shell the UI's 127.0.0.1:<port> origin changes every launch and wipes localStorage.

export const SIMPLE_MODE = "simple";
export const ADVANCED_MODE = "advanced";

const STORAGE_KEY = "sceneworks-simple-ui-default";

/**
 * The persisted "Use Simple UI by default" preference. `false` when unset or when
 * localStorage is unavailable, so an untouched install lands on the full workspace.
 */
export function readStoredSimpleDefault() {
  if (typeof window === "undefined") {
    return false;
  }
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

/** Cache the preference for the next instant paint. Silently no-ops when unavailable. */
export function writeStoredSimpleDefault(value) {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, value ? "true" : "false");
  } catch {
    // private mode / quota — the durable server copy still carries it.
  }
}

/**
 * Resolve the shell to render.
 * @param {object} input
 * @param {boolean} input.simpleDefault - persisted "Use Simple UI by default".
 * @param {"simple"|"advanced"|null} input.override - this session's sidebar pick.
 * @param {number|null} input.width - measured viewport width (null ⇒ unknown).
 * @returns {"simple"|"advanced"}
 */
export function resolveUiMode({ simpleDefault, override, width }) {
  if (isPhoneWidth(width)) {
    return SIMPLE_MODE;
  }
  if (override === SIMPLE_MODE || override === ADVANCED_MODE) {
    return override;
  }
  return simpleDefault ? SIMPLE_MODE : ADVANCED_MODE;
}

/**
 * True when the viewport itself is forcing Simple, so the switch must present as
 * locked rather than silently ignoring a click.
 */
export function uiModeLockedToSimple(width) {
  return isPhoneWidth(width);
}
