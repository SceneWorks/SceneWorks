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

// CROSS-TAB SYNC (sc-15165). Storage is shared across every tab on the origin but the
// access gate's React `token` state is not, so before this the two could diverge: tab A
// hits "forget", storage empties, and tab B's gate keeps holding a live token. Tab B then
// stays fully unlocked while every `readAccessToken()` caller in it (`serverToken()`,
// `uiPreferences.js`) sends "" — those writes 401 and, correctly, do not raise the gate
// (sc-15105), so they silently no-op. The mirror case is the same shape: tab A unlocks and
// tab B keeps showing the password prompt.
//
// The fix is to make this module the one live source rather than a seed the gate copies
// once. The browser's `storage` event fires only in the OTHER tabs on the origin, which is
// exactly the gap: a tab that mutates the token already updates its own state in the same
// call (`saveToken` / `lockRemote`), so there is deliberately NO local notify here.
//
// That is not just an optimization — a local notify would be actively wrong today.
// `saveToken` calls `storeAccessToken(candidate)` BEFORE `setToken(candidate)`, and the
// gate's `tokenRef` only re-syncs after commit, so a synchronous notify would compare the
// new token against the stale ref, fail the "unchanged" guard, and demote a just-accepted
// token back to "pending" — an extra verify POST and a shell that re-blocks for a beat.
//
// PRECONDITION: this holds only while `useAccessGate` is the sole subscriber, because it
// is the thing performing those local writes. A second subscriber (a preferences cache, a
// second React root) would silently miss every local write; adding one means making
// `storeAccessToken`/`clearAccessToken` notify, which requires inverting the
// write-before-setState ordering in `saveToken` first.
const subscribers = new Set();
let listening = false;

function handleStorageEvent(event) {
  // `key === null` is a whole-store `clear()`, which affects our key too.
  if (event.key !== null && event.key !== ACCESS_TOKEN_KEY) {
    return;
  }
  // Ignore a same-origin frame writing this key to sessionStorage. A synthetic event with
  // no `storageArea` is let through (real browsers always set it), and so is one that
  // arrives while our own store is unreachable — there is nothing to compare it against.
  const area = storage();
  if (area && event.storageArea && event.storageArea !== area) {
    return;
  }
  // Read through the helper rather than trusting `event.newValue`: subscribers must be
  // handed exactly what every other reader in the tab will see at this moment.
  const next = readAccessToken();
  // Snapshot, then re-check membership per call. A subscriber added during delivery does
  // NOT see the in-flight event (it wasn't listening when it happened) and one removed
  // during delivery does NOT get called (an unmounting hook must go quiet immediately) —
  // neither of which plain `for...of` over the live Set gives you.
  for (const listener of [...subscribers]) {
    if (!subscribers.has(listener)) {
      continue;
    }
    try {
      listener(next);
    } catch {
      // One bad subscriber must not strand the others (or the listener itself).
    }
  }
}

// Observe token changes made in OTHER tabs. Returns an unsubscribe. The window listener is
// attached on the first subscriber and detached with the last, so a non-browser or
// fully-unmounted context holds nothing.
export function subscribeAccessToken(listener) {
  subscribers.add(listener);
  if (!listening && typeof window !== "undefined") {
    window.addEventListener("storage", handleStorageEvent);
    listening = true;
  }
  return () => {
    subscribers.delete(listener);
    if (listening && subscribers.size === 0 && typeof window !== "undefined") {
      window.removeEventListener("storage", handleStorageEvent);
      listening = false;
    }
  };
}
