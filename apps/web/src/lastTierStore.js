// Persistent, per-(screen, modelId) sticky for the last EXPLICIT quant tier a user picked in a
// tiered studio surface (epic 10721 / sc-10727 — the "anti-friction core"). When a user picks a
// tier for a model on a screen, that pick becomes the default the next time they open that model on
// that screen, so they stop re-selecting a tier for every generation.
//
// Keyed by (screen, modelId) and DELIBERATELY project-independent: the quant tier is a
// hardware/quality tradeoff (RAM vs. fidelity), not a per-project authoring choice, so picking q4
// for a model on Image Studio makes q4 that model's Image-Studio default in every workspace until
// the user explicitly picks another tier. This is why the sticky lives in its own store rather than
// the per-workspace `useStudioSettings` blob — the workspace dimension would wrongly reset the
// preference when switching projects.
//
// Reuses the app's existing per-UI-pref persistence pattern — a single JSON blob in localStorage
// behind a try/catch guard, mirroring hooks/useStudioSettings.js — rather than inventing a new
// mechanism. One key holds a nested { [screen]: { [modelId]: tier } } map, so screens and models
// are independent namespaces (no key-separator collisions).
//
// PRECEDENCE (epic-locked): an explicit same-session pick > this per-(screen,model) sticky > the
// hardcoded q8 base default in `defaultTierSelection` > clamp-to-installed. This module owns only
// the sticky rung: callers read it and pass the result as the `lastUsed` argument to
// `defaultTierSelection`, which honors it whenever the sticky tier is still installed and otherwise
// falls through to the base default — so a stale/uninstalled sticky is safely ignored (clamped).

import { apiFetch } from "./api.js";

const STORAGE_KEY = "sceneworks-last-tier";

// Persist the full `{ [screen]: { [modelId]: tier } }` map to the durable server copy (epic 10721 R1).
// localStorage alone does NOT survive a desktop relaunch — the shell's `127.0.0.1:<port>` origin changes
// every launch, wiping origin-keyed storage — so a pick would be forgotten each session without this.
// Best-effort + fire-and-forget: the localStorage write already succeeded, so a failed PUT only means the
// pick isn't durable this once. Public route, empty token — mirrors the global default-quality PUT.
function persistToServer(map) {
  apiFetch("/api/v1/ui-preferences", "", {
    method: "PUT",
    body: JSON.stringify({ perModelTier: map }),
  }).catch(() => {});
}

// Seed the localStorage instant-paint cache from the durable server copy on launch (App.jsx, after
// GET /api/v1/ui-preferences). Without this, `readLastTier` would miss a pick made in a previous session
// once the desktop origin — and its localStorage — changed. Overwrites the cache with the server map,
// which is authoritative across launches (the cache is only ever a within-session mirror of it).
export function seedLastTiersFromServer(map) {
  if (map && typeof map === "object") {
    writeAll(map);
  }
}

function readAll() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    // localStorage unavailable (private mode, quota, non-DOM env) — no sticky this session.
    return {};
  }
}

function writeAll(map) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch {
    // localStorage unavailable — the sticky just won't persist this session.
  }
}

// The last explicit tier the user picked for (screen, modelId), or null when none is stored. Reads
// localStorage fresh on every call (no in-memory cache), so a value written in a previous session
// is returned after an app restart.
export function readLastTier(screen, modelId) {
  if (!screen || !modelId) {
    return null;
  }
  const perScreen = readAll()[screen];
  const value = perScreen && typeof perScreen === "object" ? perScreen[modelId] : undefined;
  return typeof value === "string" && value ? value : null;
}

// Record the user's explicit tier pick for (screen, modelId), persisting immediately. A falsy
// screen/modelId/tier is ignored (nothing to key on / clearing is not a supported operation here).
export function writeLastTier(screen, modelId, tier) {
  if (!screen || !modelId || !tier) {
    return;
  }
  const map = readAll();
  const perScreen = map[screen] && typeof map[screen] === "object" ? map[screen] : {};
  if (perScreen[modelId] === tier) {
    return;
  }
  map[screen] = { ...perScreen, [modelId]: tier };
  writeAll(map);
  // Mirror the pick to the durable server copy so it survives a desktop relaunch (localStorage doesn't).
  persistToServer(map);
}
