// Single seam for the remote-access token (epic 4484 / sc-8880). The token IS the
// host access password: a correct password verified against /api/v1/auth/verify is
// promoted to the live token, sent as the Bearer credential on every authed request.
//
// The live value is held in module state and `localStorage` is its persistence cache
// (sc-15223 — see the note above `liveToken` below); the threat model of that cache is
// unchanged and is what follows.
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

// IN-MEMORY LIVE VALUE (sc-15223). `localStorage` is this module's PERSISTENCE CACHE, not
// its source of truth: it exists so the token survives a reload, and it is allowed to fail.
//
// Every write helper below swallows a storage rejection by design — private mode, disabled
// site storage, an over-quota origin — because refusing to run in those browsers would be a
// worse outcome than not persisting. But before this the read helper went straight back to
// storage, so a swallowed write left the two halves of the app disagreeing about the same
// credential for the whole session: `useAccessGate` opened the gate on a token the host had
// just proven, while every `readAccessToken()` caller (`serverToken()`, `uiPreferences.js`)
// read the empty store and sent "". Those writes 401'd, and `apiFetch` correctly does NOT
// route a credential-less 401 to the gate (sc-15105), so they silently no-op'd. Unlike the
// cross-tab case (sc-15165) no `storage` event can ever repair it — storage is never going
// to hold the value.
//
// So the live value lives here and the helpers keep it authoritative unconditionally:
// `storeAccessToken`/`clearAccessToken` set it BEFORE attempting to persist, and
// `readAccessToken()` prefers it whenever this session has set one. Persistence becomes a
// best-effort side effect, which is what it always actually was.
//
// `null` means "no helper has run this session" — distinct from `""` ("this session forgot
// the token"), which must NOT fall back to a stale stored value. A fresh page load starts at
// `null` and reads through to storage, which is the reload path the cache exists for.
let liveToken = null;

// Read the persistence cache. `null` (not `""`) when the store is unreachable or the read
// itself throws, so callers can tell "storage says empty" from "storage says nothing" — the
// difference between adopting a value and clobbering the live one with a phantom clear.
function readStoredToken() {
  try {
    const area = storage();
    return area ? (area.getItem(ACCESS_TOKEN_KEY) ?? "") : null;
  } catch {
    return null;
  }
}

// The live access token, or "" when there is none. Read at CALL time by design — the token
// is promoted mid-session when the user answers the gate, so a value cached by an importer
// would leave that session's writes unauthenticated (see `uiPreferences.js`).
export function readAccessToken() {
  return liveToken ?? readStoredToken() ?? "";
}

// Promote the verified access token to the live value, then persist it so it survives
// reloads (see threat-model note). The order matters: the live value is set unconditionally
// so a store that refuses the write cannot desync the session (sc-15223).
export function storeAccessToken(token) {
  // Coerced because `null`/`undefined` is this module's "no helper has run" sentinel: storing
  // one would silently mean "read through to storage" instead of "the token is this".
  liveToken = token ?? "";
  try {
    storage()?.setItem(ACCESS_TOKEN_KEY, token);
  } catch {
    // Storage may reject writes in private/restricted WebKit contexts.
  }
}

// Forget the token (the "lock"/forget affordance re-shows the login gate). Same ordering
// rule, and the higher-consequence half of it: a store that refuses `removeItem` must not
// leave a forgotten password live for `readAccessToken()` to keep sending.
export function clearAccessToken() {
  liveToken = "";
  try {
    storage()?.removeItem(ACCESS_TOKEN_KEY);
  } catch {
    // Treat an unavailable store as already empty.
  }
}

// Test seam: drop the in-memory value so the next `readAccessToken()` reads through to
// storage again — i.e. simulate the fresh page load that resets it in a real browser.
//
// WHO NEEDS IT: a test file that seeds sessions with a raw
// `localStorage.setItem(ACCESS_TOKEN_KEY, ...)` AND contains any test that moves the live
// value (mounting the gate is enough — `saveToken`/`lockRemote`/the re-verify rejection all
// call the helpers above). Module state outlives an individual test, so `localStorage.clear()`
// alone leaves the earlier value in place and the raw seed is then ignored. A file that only
// reads the token needs nothing, and one that calls `vi.resetModules()` per test (e.g.
// `screens/SettingsScreen.test.jsx`) already gets a fresh module instance.
export function resetAccessTokenForTests() {
  liveToken = null;
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
// write-before-setState ordering in `saveToken` first. (sc-15223 does NOT relax this: a
// local write moves `liveToken`, so a non-subscribing reader like `serverToken()` sees it
// immediately — but a SUBSCRIBER still learns nothing without a notify.)
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
  // Another tab changed the shared store, so storage is the truth again and must overwrite
  // this tab's live value (sc-15223) — otherwise a session that had set one would pin it and
  // go deaf to every cross-tab change, undoing sc-15165. That deliberately includes clobbering
  // a live token the store itself never accepted: a real `storage` event means a peer tab DID
  // write the shared store, so its value outranks ours, and a whole-store `clear()` locking us
  // is the fail-closed direction.
  //
  // The `null` guard is narrower than that, and covers only an event arriving while OUR store
  // is unreachable or its reads throw — no browser dispatches one in that state, so it is a
  // synthetic/misdelivered event with no truth to adopt. Note this is NOT the private-mode
  // case this story is about: there `getItem` works and returns null-for-absent, which reads
  // as "" and does overwrite. Only a store we cannot read at all yields `null` here.
  const stored = readStoredToken();
  if (stored !== null) {
    liveToken = stored;
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
