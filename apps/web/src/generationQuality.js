// Global "default generation quality" setting (epic 10721 / sc-10728): the app-wide baseline quant
// tier new generations use when the user hasn't stickied a tier for a specific model. Set once in
// Settings, applied everywhere as precedence rung 3 — below the per-(screen,model) sticky
// (lastTierStore) and above clamp-to-installed. It replaces the old hardcoded q8 base fallback in
// `defaultTierSelection`, which now reads it via `options.defaultQuality`.
//
// Persistence reuses the app's per-UI-pref pattern — a single value in localStorage behind a
// try/catch guard, mirroring the theme/accent helpers in appHelpers.js — rather than the
// (screen,model)-keyed lastTierStore (that store answers "which tier did you last pick FOR THIS
// MODEL"; this answers the app-wide baseline, one dimension, no keying).
//
// Vocabulary is bf16|q8|q4 (the three user-facing quality tiers); default q8, matching the worker's
// generation default (sc-10726). int8-convrot is deliberately NOT a global-default option — it is a
// candle-only niche tier, not a sensible app-wide baseline, and would be filtered out downstream.

import { DEFAULT_GENERATION_QUALITY, GENERATION_QUALITY_TIERS } from "./quantTier.js";

// Re-exported so consumers of the setting (Settings UI, studios) get the vocabulary + default from one
// import without also reaching into quantTier.js.
export { DEFAULT_GENERATION_QUALITY, GENERATION_QUALITY_TIERS } from "./quantTier.js";

const STORAGE_KEY = "sceneworks-default-generation-quality";

// Legible Settings labels, keyed by tier. Unknown keys fall back to the raw key.
const LABELS = {
  bf16: "High fidelity (bf16)",
  q8: "Balanced (Q8)",
  q4: "Fast (Q4)",
};

export function generationQualityLabel(value) {
  return LABELS[value] ?? value;
}

// The valid quality tier for `value`, or the q8 default when it isn't one of bf16|q8|q4.
export function normalizeGenerationQuality(value) {
  return GENERATION_QUALITY_TIERS.includes(value) ? value : DEFAULT_GENERATION_QUALITY;
}

// The persisted global default generation quality, or q8 when unset / invalid / localStorage is
// unavailable (private mode, quota, non-DOM env). Reads fresh on every call (no in-memory cache) so a
// value written in a previous session survives a restart and a change made in Settings is picked up the
// next time a studio derives a default tier.
export function readDefaultGenerationQuality() {
  try {
    return normalizeGenerationQuality(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return DEFAULT_GENERATION_QUALITY;
  }
}

// Persist `value` (normalized to a valid tier) as the global default and return what was stored, so the
// caller can reflect the canonical value in its UI state.
export function writeDefaultGenerationQuality(value) {
  const next = normalizeGenerationQuality(value);
  try {
    window.localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // localStorage unavailable — the setting just won't persist this session.
  }
  return next;
}
