// Single seam for the remote-access token (epic 4484 / sc-8880). The token IS the
// host access password: a correct password verified against /api/v1/auth/verify is
// promoted to the live token, sent as the Bearer credential on every authed request.
//
// STORAGE / THREAT MODEL (sc-8880, F-078): the token is persisted verbatim in
// localStorage under a single key so it survives reloads — a hard requirement for a
// LAN remote-access tool where re-typing the password every session would be a real
// UX regression (sessionStorage would force exactly that). The plaintext-at-rest
// exposure is an XSS-exfiltration risk on the app origin, but it is accepted under
// this deployment's threat model:
//   - The host binds loopback/LAN only; there is no public origin to phish.
//   - The app has a strong XSS posture (no dangerouslySetInnerHTML on host data, CSP).
//   - The token is scoped to a single self-hosted host the user already controls.
// The real XSS mitigation is an httpOnly-cookie exchange, which is an architectural
// change to the epic-4484 auth seam (server-set cookie + CSRF) rather than a client
// tweak; it is deliberately out of scope here. Keeping the key + access in one module
// means any future hardening (session vs local storage, cookie exchange) is a
// one-file change instead of hunting scattered `localStorage.getItem("sceneworks-token")`
// literals across App.jsx and credentials.js.

// The localStorage key. Do not inline this string elsewhere — import it (or the
// helpers below) so the storage contract stays centralized.
export const ACCESS_TOKEN_KEY = "sceneworks-token";

// Whether a Web Storage backend is reachable (guards non-browser / private-mode /
// storage-disabled environments where the getter can throw).
function storage() {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

// The persisted access token, or "" when none is stored / storage is unavailable.
export function readAccessToken() {
  try {
    return storage()?.getItem(ACCESS_TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

// Persist the verified access token so it survives reloads (see threat-model note).
export function storeAccessToken(token) {
  try {
    storage()?.setItem(ACCESS_TOKEN_KEY, token);
  } catch {
    // Storage may reject writes in private/restricted WebKit contexts.
  }
}

// Forget the stored token (the "lock"/forget affordance re-shows the login gate).
export function clearAccessToken() {
  try {
    storage()?.removeItem(ACCESS_TOKEN_KEY);
  } catch {
    // Treat an unavailable store as already empty.
  }
}
