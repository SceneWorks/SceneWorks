export const UI_MODE_STORAGE_KEY = "sceneworks-ui-mode";
export const ADVANCED_UI_MODE = "advanced";
export const SIMPLE_UI_MODE = "simple";
export const UI_MODES = new Set([ADVANCED_UI_MODE, SIMPLE_UI_MODE]);

export function normalizeUiMode(value) {
  return UI_MODES.has(value) ? value : ADVANCED_UI_MODE;
}

export function readStoredUiMode() {
  if (typeof window === "undefined") {
    return ADVANCED_UI_MODE;
  }
  try {
    return normalizeUiMode(window.localStorage.getItem(UI_MODE_STORAGE_KEY));
  } catch {
    return ADVANCED_UI_MODE;
  }
}

export function persistUiMode(mode) {
  const normalized = normalizeUiMode(mode);
  try {
    window.localStorage.setItem(UI_MODE_STORAGE_KEY, normalized);
  } catch {
    // localStorage unavailable; keep the in-memory mode for this session.
  }
  return normalized;
}
