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
//
// `ready` gates the live writer during the initial restart-restore/settle window
// (sc-11962): a studio restores its snapshot at mount, then the async model / LoRA /
// preset catalogs resolve. Until they do, a stray defaults-reset could momentarily
// overwrite a restored value — and this writer would persist that transient, making
// the clobber permanent. While `ready` is false we leave the stored snapshot untouched
// (it already holds the restored values); the studio flips `ready` true once its
// catalog has loaded, at which point the current (settled) state is persisted normally.
export function useStudioSettingsWriter(studio, workspaceId, settings, ready = true) {
  const serialized = JSON.stringify(settings);
  useEffect(() => {
    if (!ready) {
      return;
    }
    try {
      window.localStorage.setItem(storageKey(studio, workspaceId), serialized);
    } catch {
      // localStorage unavailable — settings just won't persist this session.
    }
  }, [studio, workspaceId, serialized, ready]);
}
