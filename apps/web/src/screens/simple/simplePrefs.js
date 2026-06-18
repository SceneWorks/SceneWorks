// Per-control "Make my default" persistence for Simple mode. A dropdown change
// is session-only; the user pins a value as their default explicitly, so a
// one-off tweak (e.g. "just 1 this time") never silently becomes permanent.
// Values are stored as strings (selects deal in strings); callers coerce.
const PREFIX = "sceneworks-simple-pref-";

export function readPref(key, fallback = null) {
  if (typeof localStorage === "undefined") return fallback;
  try {
    const value = localStorage.getItem(PREFIX + key);
    return value === null ? fallback : value;
  } catch {
    return fallback;
  }
}

export function writePref(key, value) {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(PREFIX + key, String(value));
  } catch {
    // Private mode / quota — the control still works for the session.
  }
}
