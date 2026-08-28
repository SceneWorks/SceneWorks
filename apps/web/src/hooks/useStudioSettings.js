import { useEffect, useRef } from "react";
import { persistNavigationPreferences } from "../uiPreferences.js";

// Per-workspace "last used" settings for the Image / Video studios, so leaving a
// studio and coming back restores the prompt, model, resolution, length, advanced
// options, LoRAs and preset exactly as they were — scoped to the active workspace.
const PREFIX = "sceneworks-studio";

// Studios that participate. Needed because the durable copy is ONE map keyed by workspace
// (`{ [workspaceId]: { [studio]: {…} } }`) while the cache is one localStorage key per
// (studio, workspace) — seeding has to know which keys to write, and there is no way to
// enumerate them from the map alone without trusting its contents.
const STUDIOS = ["image", "video", "audio", "character", "editor-video"];

// Kept OUT of the durable copy (sc-15425). Both are unbounded free text — a batch sheet can
// hold hundreds of prompts — and `ImageStudio` alone persists ~47 fields, so including them
// would push a workspace past the server's per-entry budget, at which point the entry is
// dropped WHOLE and the user's whole studio silently fails to restore. They stay in the
// localStorage cache, which is what actually serves within-session navigation; the contract is
// "your setup comes back after a relaunch, an in-progress batch sheet doesn't".
const NON_DURABLE_FIELDS = ["batchPromptsText", "batchVariableValues"];

const DEFAULT_WORKSPACE = "default";
// Keep in sync with MAX_STUDIO_WORKSPACES in apps/rust-api/src/preferences.rs.
const MAX_WORKSPACES = 24;

function workspaceKey(workspaceId) {
  return workspaceId ?? DEFAULT_WORKSPACE;
}

function storageKey(studio, workspaceId) {
  return `${PREFIX}-${studio}-${workspaceKey(workspaceId)}`;
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

// `loadStudioSettings()` intentionally treats an absent entry like an empty snapshot
// for callers that just need defaults. The writer additionally needs to know whether
// an empty snapshot is actually cached: an absent cache must still establish the
// durable copy on its first ready mount.
function hasCachedStudioSettings(studio, workspaceId) {
  try {
    return window.localStorage.getItem(storageKey(studio, workspaceId)) !== null;
  } catch {
    return false;
  }
}

/**
 * Seed the localStorage cache from the durable server copy on launch (App.jsx, after
 * GET /api/v1/ui-preferences).
 *
 * Without this the advanced studios lose everything on a desktop relaunch: the shell's
 * `127.0.0.1:<port>` origin changes each launch and takes origin-keyed storage with it, so the
 * cache `loadStudioSettings` reads is empty even though the user never cleared anything.
 * The server map is authoritative across launches; the cache only ever mirrors it.
 *
 * Merges per (studio, workspace) rather than replacing: the non-durable batch fields live ONLY
 * in the cache, so a blind overwrite would drop an in-progress batch sheet on every launch of a
 * session that hadn't yet re-saved it.
 *
 * @returns {number} how many (studio, workspace) snapshots were written. The caller remounts the
 *   kept-alive studios only when this is non-zero: a studio the user opened BEFORE the GET landed
 *   read an empty cache and, being keep-alive, would never re-read on its own. Remounting
 *   unconditionally would instead discard whatever they had typed in that window for nothing.
 */
export function seedStudioSettingsFromServer(map) {
  if (!map || typeof map !== "object") {
    return 0;
  }
  let written = 0;
  for (const [workspace, perStudio] of Object.entries(map)) {
    if (!perStudio || typeof perStudio !== "object") {
      continue;
    }
    for (const studio of STUDIOS) {
      const restored = perStudio[studio];
      if (!restored || typeof restored !== "object") {
        continue;
      }
      const cached = loadStudioSettings(studio, workspace);
      writeCache(studio, workspace, { ...cached, ...restored });
      written += 1;
    }
  }
  return written;
}

function writeCache(studio, workspaceId, settings) {
  try {
    window.localStorage.setItem(storageKey(studio, workspaceId), JSON.stringify(settings));
  } catch {
    // localStorage unavailable — settings just won't persist this session.
  }
}

// Rebuild the whole `{ [workspaceId]: { [studio]: {…} } }` map from the cache, minus the
// non-durable fields, and send it. The server replaces the field wholesale (it does not
// deep-merge), so the payload has to be the full map — same contract as `perModelTier`.
function persistToServer(touchedWorkspace) {
  const map = {};
  let keys;
  try {
    keys = Object.keys(window.localStorage);
  } catch {
    return;
  }
  for (const key of keys) {
    if (!key.startsWith(`${PREFIX}-`)) {
      continue;
    }
    const studio = STUDIOS.find((entry) => key.startsWith(`${PREFIX}-${entry}-`));
    if (!studio) {
      continue;
    }
    const workspace = key.slice(`${PREFIX}-${studio}-`.length);
    const settings = loadStudioSettings(studio, workspace);
    const durable = { ...settings };
    for (const field of NON_DURABLE_FIELDS) {
      delete durable[field];
    }
    map[workspace] = { ...(map[workspace] ?? {}), [studio]: durable };
  }
  // Bound the map the same way the server does, keeping the workspace being written.
  const others = Object.keys(map).filter((key) => key !== touchedWorkspace);
  const dropped = others.slice(0, Math.max(0, others.length - (MAX_WORKSPACES - 1)));
  for (const key of dropped) {
    delete map[key];
  }
  // Fire-and-forget: the cache write already succeeded, so a failed PUT costs only durability
  // across the next relaunch. Routed through the coalescing, token-attaching seam (sc-15136).
  persistNavigationPreferences({ advancedStudio: map }).catch(() => {});
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
//
// `ready` now ALSO has to cover ui-preferences hydration (sc-15425), for the same reason one
// step further out: on a relaunch the cache is empty until the GET lands, so a studio the user
// opened before then would persist its catalog defaults over the durable copy before anything
// had read it. Callers pass `preferencesHydrated && <their catalog gate>`.
export function useStudioSettingsWriter(studio, workspaceId, settings, ready = true) {
  const serialized = JSON.stringify(settings);
  // A studio normally mounts from the same cache it is about to write. Track that
  // first settled pass so it does not issue a redundant durable PUT, while retaining
  // the important ready=false -> true transition as a deliberate first persistence.
  const scopeRef = useRef("");
  const initializedRef = useRef(false);
  useEffect(() => {
    const scope = `${studio}:${workspaceKey(workspaceId)}`;
    if (scopeRef.current !== scope) {
      scopeRef.current = scope;
      initializedRef.current = false;
    }
    if (!ready) {
      initializedRef.current = true;
      return undefined;
    }
    const unchangedInitialSnapshot = !initializedRef.current
      && hasCachedStudioSettings(studio, workspaceId)
      && JSON.stringify(loadStudioSettings(studio, workspaceId)) === serialized;
    initializedRef.current = true;
    if (unchangedInitialSnapshot) {
      return undefined;
    }
    writeCache(studio, workspaceId, JSON.parse(serialized));
    // Debounced: the advanced studios re-serialize on every keystroke, and one durable write
    // per character would hammer a disk-writing PUT. The queue keeps requests ordered; this
    // keeps them rare.
    const timer = setTimeout(() => persistToServer(workspaceKey(workspaceId)), 400);
    return () => clearTimeout(timer);
  }, [studio, workspaceId, serialized, ready]);
}
