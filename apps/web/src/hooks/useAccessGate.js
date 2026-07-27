import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  apiFetch,
  isReauthenticating,
  isThrottled,
  setMediaTicket,
  setUnauthorizedHandler,
} from "../api.js";
import { isDesktop as isDesktopShell } from "../runtime.js";
import {
  readAccessToken,
  storeAccessToken,
  clearAccessToken,
  subscribeAccessToken,
} from "../accessToken.js";
import { recordStartupMark } from "../startupTiming.js";

// Owns the remote-access gate (epic 4484): the /api/v1/access probe, the host
// password (= API token), the login-gate draft/error, and the media-ticket mint
// that must settle before protected data loads. Extracted verbatim from App.jsx
// (sc-9750, F-052 follow-up) so App no longer carries the ~5 auth state slices plus
// the mint effect inline; the returned values thread through App's remaining effects
// (the [ready, token] data load, the project-switch load, the SSE stream) and the
// login-gate JSX unchanged. Behavior-preserving — same effects, deps, and cleanup.
//
// `setError`, `pushNotice`, and `dismissNoticeKind` are App-owned (they write the
// shared notices store) and passed in identity-stable, so the mint effect's deps
// stay stable across App's SSE-driven re-renders.
export function useAccessGate({ setError, pushNotice, dismissNoticeKind }) {
  const [access, setAccess] = useState({ authRequired: false });
  // Whether GET /api/v1/access has answered yet. Until it does we don't know if a
  // password is required, so a remote browser must hold its protected data loads —
  // otherwise they fire optimistically, 401, and bury the password prompt under a band
  // of "access token required" errors (epic 4484).
  const [accessResolved, setAccessResolved] = useState(false);
  // Whether the probe has failed at least once. Drives the gate's "awaiting-host"
  // state: we can't show a password prompt (we don't know one is needed) but we must
  // not hand over a navigable, permanently empty shell either (sc-15102).
  const [accessProbeFailed, setAccessProbeFailed] = useState(false);
  // The remote-access token (= host password). Storage key + threat-model note live in
  // accessToken.js (sc-8880); it is persisted verbatim in localStorage so it survives
  // reloads (a LAN remote-access requirement — see that module for the accepted XSS tradeoff).
  const [token, setToken] = useState(() => readAccessToken());
  // Whether `token` has been PROVEN against the host: "none" (no token) / "pending"
  // (restored from localStorage, not yet checked) / "accepted".
  //
  // sc-15102: a token restored from storage used to be trusted on sight, so a password
  // the host has since changed or cleared flipped `authenticated` true and every
  // protected request 401'd — with no code path anywhere in the client that re-prompts
  // on a 401. The result was an app that looked broken instead of locked, escapable only
  // by finding the topbar "Lock" button. Now the restored token is re-verified before it
  // counts, and a rejected one is dropped back to the password prompt.
  const [tokenStatus, setTokenStatus] = useState(() => (readAccessToken() ? "pending" : "none"));
  // What the user is typing into the login gate (sc-8808). Kept separate from the
  // live `token` so keystrokes never flip `authenticated` or churn the data/SSE
  // effects; `token` only changes once /api/v1/auth/verify accepts the draft.
  const [passwordDraft, setPasswordDraft] = useState("");
  // Wrong-password feedback for the remote-browser login gate (epic 4484 story 7).
  const [authError, setAuthError] = useState("");
  // sc-15105: mirrors of the two values the 401 handler below reads. It runs outside
  // React's render cycle (inside `apiFetch`) and must decide before any state has
  // flushed, so it cannot read `tokenStatus`/`access` directly. Synced in a bare
  // useLayoutEffect (no dep array) rather than the render body — a discarded concurrent
  // or StrictMode render must not be able to publish state that never committed, and
  // must not clobber the eager write `handleUnauthorized` makes below (the same rule
  // App.jsx follows for its own live-value refs, sc-8940 / sc-9641).
  const tokenStatusRef = useRef(tokenStatus);
  const authRequiredRef = useRef(false);
  // sc-15165: the cross-tab subscriber below runs outside render too, and has to compare
  // the incoming storage value against the live token to decide whether anything changed.
  const tokenRef = useRef(token);
  useLayoutEffect(() => {
    tokenStatusRef.current = tokenStatus;
    authRequiredRef.current = access.authRequired === true;
    tokenRef.current = token;
  });

  // The desktop shell reaches its own API over loopback, which the API trusts
  // (SCENEWORKS_TRUST_LOOPBACK), so it's authenticated without a password — never prompt
  // for one locally (epic 4484). A remote browser must wait for GET /api/v1/access before
  // it knows whether a password is needed; until then it holds its protected loads rather
  // than firing them unauthenticated.
  const authenticated = useMemo(
    () =>
      isDesktopShell ||
      (accessResolved && (!access.authRequired || tokenStatus === "accepted")),
    [accessResolved, access, tokenStatus],
  );
  // What the full-page blocker should show, if anything (sc-15102). "open" means the app
  // shell may render; every other value blocks it. The policy lives here, next to the
  // state it reads, so App only has to branch on `gateStatus !== "open"`.
  const gateStatus = useMemo(() => {
    if (isDesktopShell) {
      // Loopback-trusted: the desktop shell is never gated (epic 4484).
      return "open";
    }
    if (!accessResolved) {
      // The auth requirement is still unknown, so no protected screen may mount.
      // A failed probe changes the copy/notice, but both states remain full-page
      // blockers until the host answers.
      return accessProbeFailed ? "awaiting-host" : "probing";
    }
    if (!access.authRequired) {
      return "open";
    }
    if (tokenStatus === "accepted") {
      return "open";
    }
    return tokenStatus === "pending" ? "unlocking" : "locked";
  }, [access, accessProbeFailed, accessResolved, tokenStatus]);
  // sc-8810: whether media URLs are renderable yet. When auth is on, element-driven
  // requests (<img>/<video>) can't send the token header, so every media URL needs
  // the query-param ticket minted below. Data loads hold until the first ticket is
  // stored (mediaReady), otherwise thumbnails rendered in the gap would 401 and
  // stick as "deleted" placeholders. When auth is off this resolves immediately.
  const [mediaReady, setMediaReady] = useState(false);
  // sc-9063 (F-008 follow-up): a persistently failing mint must not blank the whole
  // app. `ready` waits for the first mint attempt to SETTLE (success or failure),
  // not for a success: once a mint fails, lists/metadata load anyway — media
  // degrades to placeholders and recovers when a retry lands — and a "media-ticket"
  // notice tells the user why media is broken while the backoff keeps retrying.
  const [mediaTicketFailed, setMediaTicketFailed] = useState(false);
  const ready = authenticated && (mediaReady || mediaTicketFailed);

  useEffect(() => {
    if (!authenticated || !accessResolved) {
      // Lock/logout stops the mint loop (cleanup above cleared the backoff timer),
      // so drop the settled gate: the next unlock must re-run sc-8810's
      // mint-before-data ordering rather than ride a stale mediaReady/
      // mediaTicketFailed, and the "Retrying in the background" notice must not
      // linger on the lock screen while no retry is actually running.
      setMediaReady(false);
      setMediaTicketFailed(false);
      dismissNoticeKind("media-ticket");
      return undefined;
    }
    if (!access.authRequired) {
      setMediaTicket("");
      recordStartupMark("media-authorization-settled");
      setMediaReady(true);
      return undefined;
    }
    let closed = false;
    let timer = null;
    let attempt = 0;
    async function acquire() {
      try {
        // Header-authenticated mint (loopback-trusted desktop sends no token and
        // still passes). The server keeps re-arming the same sliding ticket, so
        // already-rendered media URLs stay valid across refreshes.
        const response = await apiFetch("/api/v1/files/ticket", token, { method: "POST" });
        if (closed) {
          return;
        }
        setMediaTicket(response.ticket);
        recordStartupMark("media-authorization-settled");
        setMediaReady(true);
        setMediaTicketFailed(false);
        dismissNoticeKind("media-ticket");
        attempt = 0;
        const ttlMs = Math.max(15, Number(response.expiresInSeconds) || 0) * 1000;
        timer = window.setTimeout(acquire, Math.max(5000, Math.floor(ttlMs / 3)));
      } catch (err) {
        if (closed) {
          return;
        }
        if (isReauthenticating(err)) {
          // sc-15105: the token went stale under us and the gate has claimed the session.
          // Surrender instead of arming a backoff that can only 401 again forever — and
          // skip the "retrying in the background" notice, which would be a lie (nothing
          // is retrying) on the blocker that is about to appear. The effect re-runs and
          // mints afresh if the re-verify keeps the session. A 401 the gate did NOT claim
          // falls through to the ordinary degraded-media path below.
          return;
        }
        recordStartupMark("media-authorization-settled");
        setMediaTicketFailed(true);
        pushNotice(
          "media-ticket",
          "media authorization: couldn't obtain a media ticket, so thumbnails, previews, and downloads may fail to load. Retrying in the background.",
        );
        const delay = Math.min(30000, 1000 * 2 ** attempt);
        attempt += 1;
        timer = window.setTimeout(acquire, delay);
      }
    }
    acquire();
    return () => {
      closed = true;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, [access.authRequired, accessResolved, authenticated, token, pushNotice, dismissNoticeKind]);

  // Probe whether the deployment requires a password, then release `accessResolved`
  // so an authenticated client (or one not requiring auth) can load its data. Mirrors
  // the health fetch's mount timing but owns only the access state (App keeps health).
  //
  // sc-11231 (F-038): the gate releases ONLY on a successful probe. The prior code did
  // `.finally(() => setAccessResolved(true))`, so a single failed GET /api/v1/access
  // released the gate while `access` stayed `{ authRequired: false }` → `authenticated`
  // flipped true with no token, protected data + SSE fired unauthenticated, and the login
  // band (which needs `access.authRequired`) never rendered — on an auth-required remote
  // deployment that is a 401 storm with no login path, recoverable only by reload (fail
  // OPEN). Instead we hold the gate closed and retry with exponential backoff — mirroring
  // the media-ticket mint loop above — so a transient failure self-heals and a persistent
  // outage surfaces an "access-probe" notice rather than silently authenticating.
  useEffect(() => {
    let closed = false;
    let timer = null;
    let attempt = 0;
    async function probe() {
      try {
        const result = await apiFetch("/api/v1/access", "");
        if (closed) {
          return;
        }
        setAccess(result);
        recordStartupMark("access-resolved");
        setAccessResolved(true);
        setAccessProbeFailed(false);
        dismissNoticeKind("access-probe");
      } catch {
        if (closed) {
          return;
        }
        // Keep accessResolved false (gate stays closed → no unauthenticated loads), tell
        // the user why, and schedule a backoff retry so recovery is automatic.
        setAccessProbeFailed(true);
        // No raw setError(err.message) here: the "access-probe" notice below says the same
        // thing in human terms, and on the sc-15102 blocker the two stacked as a bare
        // "Failed to fetch" above the readable one.
        pushNotice(
          "access-probe",
          "access check: couldn't reach the host to determine whether a password is required. Retrying in the background.",
        );
        const delay = Math.min(30000, 1000 * 2 ** attempt);
        attempt += 1;
        timer = window.setTimeout(probe, delay);
      }
    }
    probe();
    return () => {
      closed = true;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
    // Mount-only probe (matches the pre-extraction App effect); setError, pushNotice, and
    // dismissNoticeKind are App-owned and identity-stable, so the retry loop captures them
    // once safely.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // sc-15105: the same staleness can arrive MID-SESSION — the host rotates its password
  // and restarts the API while the tab sits open — and sc-15102's re-verify only runs at
  // startup, so nothing re-prompted. `apiFetch` now calls this on every 401.
  //
  // Deliberately NOT a logout: one unlucky endpoint must not evict a good session. We
  // demote the token from "accepted" back to "pending", which drops `authenticated`
  // (blocking the shell behind the gate's "unlocking" state and stopping the mint/SSE
  // loops via their effect cleanups) and re-arms the re-verify effect below — the single
  // place that decides between "the password really did change" (ok:false ⇒ drop it, show
  // the prompt) and "route-specific 401" (ok:true ⇒ keep the session, resume).
  //
  // Only demote from "accepted". "none" means the gate is already locked with no token —
  // flipping that to "pending" would show "unlocking" forever, because the re-verify
  // effect below bails on an empty token and nothing would ever move it on. "pending"
  // means a check is already in flight. Both are still the gate owning the failure, so
  // they return true. The eager ref write is what makes a storm collapse: a page full of
  // screens 401ing at once all land here before React flushes.
  //
  // Returns whether the gate claimed the session; `apiFetch` records that on the error as
  // `reauthenticating` so the mint / SSE retry loops can tell "surrender, the gate has
  // it" from "an ordinary failure I should keep handling myself".
  const handleUnauthorized = useCallback(() => {
    // A deployment with no password can't produce a 401 the gate could act on — a 401
    // here means the host disagrees with its own /api/v1/access answer, which is not
    // something a password prompt fixes.
    if (!authRequiredRef.current) {
      return false;
    }
    if (tokenStatusRef.current === "accepted") {
      tokenStatusRef.current = "pending";
      setTokenStatus("pending");
    }
    return true;
  }, []);

  useEffect(() => {
    if (isDesktopShell) {
      // The desktop shell reaches its own API over trusted loopback and is authenticated
      // unconditionally (see `authenticated` above) — it must never be prompted, so it
      // doesn't register a 401 reaction at all (epic 4484).
      return undefined;
    }
    return setUnauthorizedHandler(handleUnauthorized);
  }, [handleUnauthorized]);

  // Adopt a token change made in ANOTHER tab (sc-15165). `token` used to be seeded from
  // storage once at mount, so the two readers of the same credential — this state and
  // `readAccessToken()`, which `serverToken()` / `uiPreferences.js` call fresh every time —
  // could disagree for the rest of the session. Subscribing collapses them back onto one
  // value: a "forget" in tab A now locks tab B (instead of leaving it live but writing
  // credential-less), and an unlock in tab A now opens tab B (instead of stranding its
  // prompt). Storage events never fire in the tab that made the change, so this only ever
  // sees external mutations and can't race `saveToken`/`lockRemote`'s own state writes.
  //
  // An adopted token is "pending", not "accepted": it was proven against the host by the
  // OTHER tab, and this tab must clear the same sc-15102 bar as one restored from storage
  // before it authenticates — the re-verify effect below picks it up and promotes it. The
  // cost is real and accepted: "pending" means `authenticated` is false for one
  // /auth/verify round-trip, and App renders the gate INSTEAD of either shell, so an
  // already-open tab tears its tree down and back up (losing drawer state, unsent prompt
  // drafts, scroll) on a benign cross-tab unlock. Trusting the other tab's word instead
  // would re-open the exact hole sc-15102 closed — a token the host has since rotated,
  // authenticated on sight, 401ing every request behind a shell that looks merely broken.
  //
  // No `isDesktopShell` guard, unlike the effects either side: the desktop shell is
  // loopback-trusted, and both `authenticated` and `gateStatus` short-circuit on
  // `isDesktopShell` before they ever read `tokenStatus`, so a cross-tab change cannot
  // gate it. (It can leave `tokenStatus` parked at "pending" there — the re-verify effect
  // below bails on desktop and nothing else moves it — so a future `tokenStatus` branch
  // must not assume desktop reaches a terminal value.)
  useEffect(
    () =>
      subscribeAccessToken((next) => {
        if (next === tokenRef.current) {
          // A re-store of the same token (another tab re-verified it, or unlocked with the
          // same password). Nothing changed here, and demoting an accepted token to
          // "pending" over it would re-block this shell and cost a needless verify POST.
          return;
        }
        // Eager write, same rule as `handleUnauthorized` above: a burst of storage events
        // must all compare against the newest value, not the last committed render's.
        tokenRef.current = next;
        setToken(next);
        setTokenStatus(next ? "pending" : "none");
        setPasswordDraft("");
        setAuthError("");
        // The re-verify loop is about to be torn down by the token change, so its
        // "retrying in the background" notice would be a lie (same reasoning as
        // `lockRemote`).
        dismissNoticeKind("access-verify");
      }),
    [dismissNoticeKind],
  );

  // Re-verify a token restored from localStorage before trusting it (sc-15102). The host
  // password can change (or be cleared, or the host can be a different machine at the same
  // LAN address) between visits, and nothing in the client re-prompts on a 401 — so an
  // unproven stored token has to clear this check before it counts as authentication.
  // Ordering matters: this runs BEFORE `authenticated` flips, so it costs one POST ahead of
  // the first protected load and never races it. A rejected token is dropped and the gate
  // shows the password prompt; an unreachable host keeps the token and retries with
  // backoff (mirroring the probe/mint loops) so a transient outage self-heals.
  //
  // sc-15105 gave this effect a second entry point: `handleUnauthorized` above demotes an
  // accepted token to "pending" on a 401, so the same check now adjudicates both "restored
  // from storage, never proven" and "was proven, then the host rotated its password".
  useEffect(() => {
    if (isDesktopShell || !accessResolved || !access.authRequired) {
      return undefined;
    }
    if (tokenStatus !== "pending" || !token) {
      return undefined;
    }
    let closed = false;
    let timer = null;
    let attempt = 0;
    async function verify() {
      try {
        const result = await apiFetch("/api/v1/auth/verify", token, { method: "POST" });
        if (closed) {
          return;
        }
        if (result?.ok) {
          setTokenStatus("accepted");
          dismissNoticeKind("access-verify");
          return;
        }
        // Definitively rejected: forget it rather than 401 every request with it.
        clearAccessToken();
        setToken("");
        setTokenStatus("none");
        setAuthError("The saved password no longer works on this host. Enter the current one.");
        dismissNoticeKind("access-verify");
      } catch (err) {
        if (closed) {
          return;
        }
        pushNotice(
          "access-verify",
          isThrottled(err)
            ? "access check: the host is temporarily refusing password attempts from this device after too many failures. Retrying in the background."
            : "access check: couldn't reach the host to confirm the saved password. Retrying in the background.",
        );
        const delay = Math.min(30000, 1000 * 2 ** attempt);
        attempt += 1;
        timer = window.setTimeout(verify, delay);
      }
    }
    verify();
    return () => {
      closed = true;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, [access.authRequired, accessResolved, token, tokenStatus, pushNotice, dismissNoticeKind]);

  // Remote-browser login (epic 4484 story 7): the password IS the API access token.
  // Verify the typed draft against the public /api/v1/auth/verify endpoint BEFORE
  // promoting it to the live `token`, so a wrong password keeps the gate up with an
  // inline error (instead of saving a bad token and silently failing every subsequent
  // request). A correct password is stored to localStorage and unlocks the app; it
  // persists across reloads. Promoting the token here flips `authenticated`, and the
  // [authenticated, token] effects perform the initial data load and SSE connect
  // exactly once — no explicit refreshData() call, or it would double-fetch (sc-8808).
  const saveToken = useCallback(
    async (event) => {
      event.preventDefault();
      const candidate = passwordDraft.trim();
      if (!candidate) {
        setAuthError("Enter the password.");
        return;
      }
      try {
        const result = await apiFetch("/api/v1/auth/verify", candidate, { method: "POST" });
        if (!result?.ok) {
          setAuthError("Incorrect password. Try again.");
          return;
        }
      } catch (err) {
        // sc-15105: a rotation storm can trip the host's per-IP attempt throttle
        // (`apps/rust-api/src/auth.rs`), which answers 429 — including for this public
        // verify route. Reporting that as "couldn't reach the host" sends the user
        // hunting a network fault instead of waiting out a lockout that clears itself.
        setAuthError(
          isThrottled(err)
            ? "Too many password attempts from this device. Wait a minute, then try again."
            : "Couldn't reach the host to verify the password.",
        );
        return;
      }
      storeAccessToken(candidate);
      setToken(candidate);
      // Just proven against the host — mark it accepted so the sc-15102 re-verify effect
      // doesn't immediately check the same string again (which would double-POST and
      // briefly re-block the shell).
      setTokenStatus("accepted");
      setPasswordDraft("");
      setAuthError("");
      setError("");
      dismissNoticeKind("access-verify");
    },
    [passwordDraft, setError, dismissNoticeKind],
  );

  // Clear the stored password and re-show the login gate ("lock"/forget affordance,
  // epic 4484 story 7). Setting the token state to "" re-renders the gate, which
  // keys off the token state (sc-8808).
  const lockRemote = useCallback(() => {
    clearAccessToken();
    setToken("");
    setTokenStatus("none");
    setPasswordDraft("");
    setAuthError("");
    // The re-verify loop's cleanup just stopped its backoff timer, so its
    // "retrying in the background" notice would be a lie on the lock screen
    // (same reasoning as the media-ticket notice above).
    dismissNoticeKind("access-verify");
  }, [dismissNoticeKind]);

  return {
    access,
    token,
    passwordDraft,
    setPasswordDraft,
    authError,
    authenticated,
    gateStatus,
    ready,
    saveToken,
    lockRemote,
  };
}
