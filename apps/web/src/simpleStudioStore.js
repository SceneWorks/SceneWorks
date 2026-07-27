// Durable store for the Simple studios' own knobs — model, prompt, resolution, duration,
// LoRA picks (epic 15404's fields and whatever Simple grows next).
//
// The Simple shell renders one screen at a time, so a studio UNMOUNTS on navigation and its
// state dies with it. The shell keeps a session copy (see simple/useStudioState.js); this
// module is what makes that copy outlive an app relaunch.
//
// WHY NOT MIRROR THE ADVANCED STUDIOS. They solve the same problem with a per-workspace
// localStorage blob (hooks/useStudioSettings.js) plus keep-alive mounting (App.jsx's
// KEEP_ALIVE_VIEWS), so their state never has to survive a real unmount. localStorage alone
// does NOT survive a desktop relaunch — the shell's `127.0.0.1:<port>` origin changes every
// launch, wiping origin-keyed storage — which is exactly the durability the Simple studios
// need. So this follows lastTierStore.js instead: localStorage is the instant-paint cache and
// `ui-preferences.json` is authoritative across launches.
//
// KEYED BY WORKSPACE, matching the advanced studios' `sceneworks-studio-<studio>-<workspaceId>`
// namespacing. A prompt written for one project must not follow the user into another.

import { persistNavigationPreferences } from "./uiPreferences.js";

const STORAGE_KEY = "sceneworks-simple-studio";

// Keep in sync with MAX_SIMPLE_STUDIO_WORKSPACES in apps/rust-api/src/preferences.rs. Nothing
// ever removes a workspace entry, so without a bound a long-lived install would carry every
// project it has ever opened — each holding a full prompt. Pruning here rather than relying on
// the server's backstop keeps the localStorage cache and the durable copy the same shape.
const MAX_WORKSPACES = 24;

const DEFAULT_WORKSPACE = "default";

function workspaceKey(workspaceId) {
  return workspaceId ?? DEFAULT_WORKSPACE;
}

function readAll() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    // localStorage unavailable (private mode, quota, non-DOM env) — no restore this session.
    return {};
  }
}

function writeAll(map) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch {
    // localStorage unavailable — the server copy below is still attempted, so the settings
    // can come back on the next launch even when the cache can't hold them now.
  }
}

// Drop the oldest entries once the map exceeds the bound. `touched` is moved to the end first
// so the workspace the user is actually in is never the one evicted.
function prune(map, touched) {
  const keys = Object.keys(map).filter((key) => key !== touched);
  const kept = keys.slice(Math.max(0, keys.length - (MAX_WORKSPACES - 1)));
  const next = {};
  for (const key of kept) {
    next[key] = map[key];
  }
  next[touched] = map[touched];
  return next;
}

// Seed the localStorage instant-paint cache from the durable server copy on launch (App.jsx,
// after GET /api/v1/ui-preferences). Without this, a relaunched desktop app reads an empty
// cache — its origin, and therefore its localStorage, changed — and every studio would come
// back on catalog defaults. The server map is authoritative; the cache only ever mirrors it.
export function seedSimpleStudiosFromServer(map) {
  if (map && typeof map === "object") {
    writeAll(map);
  }
}

/**
 * The stored `{ [studio]: { [field]: value } }` snapshot for a workspace, or `{}` when none.
 * Read fresh from localStorage on every call, so a value seeded from the server after launch
 * is picked up without this module holding its own copy.
 */
export function readSimpleStudioSnapshot(workspaceId) {
  const entry = readAll()[workspaceKey(workspaceId)];
  return entry && typeof entry === "object" ? entry : {};
}

/**
 * Record a workspace's full studio snapshot: localStorage now, server best-effort.
 *
 * The PUT goes through `persistNavigationPreferences`, which coalesces patches and keeps at
 * most one request in flight — so a burst of typing collapses instead of racing, and an older
 * delayed response can't land on top of a newer snapshot. Callers still debounce; this is the
 * ordering guarantee, not the rate limit.
 */
export function writeSimpleStudioSnapshot(workspaceId, snapshot) {
  if (!snapshot || typeof snapshot !== "object") {
    return;
  }
  const key = workspaceKey(workspaceId);
  const map = readAll();
  map[key] = snapshot;
  const pruned = prune(map, key);
  writeAll(pruned);
  // Fire-and-forget: the cache write already succeeded, so a failed PUT costs only durability
  // across the next relaunch. The PUT is a GATED write, so it goes through the token-attaching
  // seam (sc-15136) rather than a hand-rolled credential-less apiFetch.
  persistNavigationPreferences({ simpleStudio: pruned }).catch(() => {});
}
