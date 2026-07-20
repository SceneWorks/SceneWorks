// Global "default generation quality" setting (epic 10721 / sc-10728): the app-wide baseline quant
// tier new generations use when the user hasn't stickied a tier for a specific model. Set once in
// Settings, applied everywhere as precedence rung 3 — below the per-(screen,model) sticky
// (lastTierStore) and above clamp-to-installed. It replaces the old hardcoded q8 base fallback in
// `defaultTierSelection`, which now reads it via `options.defaultQuality`.
//
// Persistence mirrors theme/accent (sc-10728): the durable copy lives server-side in
// `ui-preferences.json`, written through `PUT /api/v1/ui-preferences` by the Settings change handler
// and re-seeded from `GET /api/v1/ui-preferences` on launch (App.jsx). localStorage is only an
// instant-paint cache — the read/write helpers below touch it — because on the desktop shell the UI
// runs at the API's per-launch `http://127.0.0.1:<port>` origin, where origin-keyed localStorage does
// NOT survive a relaunch (the port, and so the origin, changes every launch). This is one app-wide
// dimension, not the (screen,model)-keyed lastTierStore ("which tier did you last pick FOR THIS
// MODEL").
//
// Vocabulary is bf16|q8|q4 (the three user-facing quality tiers); default q8, matching the worker's
// generation default (sc-10726). int8-convrot is deliberately NOT a global-default option — it is a
// candle-only niche tier, not a sensible app-wide baseline, and would be filtered out downstream.

import { DEFAULT_GENERATION_QUALITY, GENERATION_QUALITY_TIERS } from "./quantTier.js";

// Re-exported so consumers of the setting (Settings UI, studios) get the vocabulary + default from one
// import without also reaching into quantTier.js.
export { DEFAULT_GENERATION_QUALITY, GENERATION_QUALITY_TIERS } from "./quantTier.js";

// The capability-aware "Auto" mode (epic 10721 R3) — the DEFAULT. In Auto, each model's default tier is
// the highest-fidelity tier that fits this machine's memory (`suggestTier`), so a small model defaults to
// bf16 and a heavy one on a small Mac defaults to what fits — instead of a flat tier for everything. A
// user can still pin an explicit bf16/q8/q4 globally, or override per-model in a studio (that pick is
// saved). NOT a quant tier, so it is deliberately absent from `GENERATION_QUALITY_TIERS`.
export const AUTO_GENERATION_QUALITY = "auto";

// The Settings dropdown vocabulary, in display order: Auto first (the default), then the explicit tiers.
export const GENERATION_QUALITY_OPTIONS = [AUTO_GENERATION_QUALITY, ...GENERATION_QUALITY_TIERS];

const STORAGE_KEY = "sceneworks-default-generation-quality";

// Legible Settings labels, keyed by value. Unknown keys fall back to the raw key.
const LABELS = {
  [AUTO_GENERATION_QUALITY]: "Auto (best that fits this Mac)",
  bf16: "High fidelity (bf16)",
  q8: "Balanced (Q8)",
  q4: "Fast (Q4)",
};

export function generationQualityLabel(value) {
  return LABELS[value] ?? value;
}

// The valid setting for `value` — "auto" or an explicit bf16|q8|q4 — else the "auto" default when it
// isn't one of them. Auto is the app-wide default (epic 10721 R3): an unset/invalid preference means
// "let the app pick the best tier that fits", not a flat q8.
export function normalizeGenerationQuality(value) {
  return value === AUTO_GENERATION_QUALITY || GENERATION_QUALITY_TIERS.includes(value)
    ? value
    : AUTO_GENERATION_QUALITY;
}

// The global default generation quality from the instant-paint cache, or q8 when unset / invalid /
// localStorage is unavailable (private mode, quota, non-DOM env). Reads fresh on every call (no
// in-memory cache) so a value seeded from the server on launch — or changed in Settings — is picked up
// the next time a studio derives a default tier. On desktop the cache is re-seeded from the durable
// server copy at launch (App.jsx), so it reflects the previous session even after the origin changed.
export function readDefaultGenerationQuality() {
  try {
    return normalizeGenerationQuality(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return DEFAULT_GENERATION_QUALITY;
  }
}

// Write `value` (normalized to a valid tier) into the instant-paint localStorage cache and return what
// was stored. This is ONLY the cache write; the durable copy is persisted separately through
// `PUT /api/v1/ui-preferences` (see SettingsScreen.changeDefaultQuality). The launch-time seed also
// calls this to prime the cache from the server without echoing a redundant PUT back.
export function writeDefaultGenerationQuality(value) {
  const next = normalizeGenerationQuality(value);
  try {
    window.localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // localStorage unavailable — the cache just won't persist this session (the server copy still does).
  }
  return next;
}
