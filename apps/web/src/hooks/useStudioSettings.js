import { useEffect } from "react";

// Per-workspace "last used" settings for the Image / Video studios, so leaving a
// studio and coming back restores the prompt, model, resolution, length, advanced
// options, LoRAs and preset exactly as they were — scoped to the active workspace.
const PREFIX = "sceneworks-studio";

function storageKey(studio, workspaceId) {
  return `${PREFIX}-${studio}-${workspaceId ?? "default"}`;
}

// Read the saved snapshot once at mount. Returns {} when nothing is stored or
// localStorage is unavailable (private mode, quota), so callers can `?? default`.
export function loadStudioSettings(studio, workspaceId) {
  try {
    const raw = window.localStorage.getItem(storageKey(studio, workspaceId));
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

// Persist `settings` for the workspace whenever they change. Serializing on every
// render is cheap for a couple dozen fields and lets the effect skip writes when
// nothing actually changed.
export function useStudioSettingsWriter(studio, workspaceId, settings) {
  const serialized = JSON.stringify(settings);
  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey(studio, workspaceId), serialized);
    } catch {
      // localStorage unavailable — settings just won't persist this session.
    }
  }, [studio, workspaceId, serialized]);
}
