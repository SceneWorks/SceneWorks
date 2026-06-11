// Draft <-> value helpers for the training config forms (sc-4199). Extracted
// verbatim from TrainingStudio.jsx — these convert between the string-typed form
// drafts the inputs hold and the numeric/clamped values the worker payload needs.
// Pure functions, no React, no app state.

export function asText(value) {
  return value === null || value === undefined ? "" : String(value);
}

// Normalize a training-adapter version to the worker's canonical token. Mirrors
// the worker's substring match so legacy "v2-default" shows as "v2" in the select.
export function normalizeTrainingAdapterVersion(value) {
  const token = asText(value).trim();
  const lower = token.toLowerCase();
  if (lower.includes("v1")) return "v1";
  if (lower.includes("v2")) return "v2";
  return token;
}

export function numericDraft(value) {
  return value === null || value === undefined ? "" : String(value);
}

export function numberFromDraft(value) {
  const trimmed = String(value ?? "").trim();
  if (!trimmed) {
    return null;
  }
  const number = Number(trimmed);
  return Number.isFinite(number) ? number : null;
}

export function boundedNumber(value, fallback, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, number));
}

export function integerFromDraft(value, fallback, min, max) {
  return Math.round(boundedNumber(value, fallback, min, max));
}

export function compactObject(object) {
  return Object.fromEntries(
    Object.entries(object).filter(([, value]) => value !== "" && value !== null && value !== undefined),
  );
}
