import { useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch, setMediaTicket } from "../api.js";
import { isDesktop as isDesktopShell } from "../runtime.js";
import { readAccessToken, storeAccessToken, clearAccessToken } from "../accessToken.js";
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
      } catch {
        if (closed) {
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

  // Re-verify a token restored from localStorage before trusting it (sc-15102). The host
  // password can change (or be cleared, or the host can be a different machine at the same
  // LAN address) between visits, and nothing in the client re-prompts on a 401 — so an
  // unproven stored token has to clear this check before it counts as authentication.
  // Ordering matters: this runs BEFORE `authenticated` flips, so it costs one POST ahead of
  // the first protected load and never races it. A rejected token is dropped and the gate
  // shows the password prompt; an unreachable host keeps the token and retries with
  // backoff (mirroring the probe/mint loops) so a transient outage self-heals.
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
      } catch {
        if (closed) {
          return;
        }
        pushNotice(
          "access-verify",
          "access check: couldn't reach the host to confirm the saved password. Retrying in the background.",
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
      } catch {
        setAuthError("Couldn't reach the host to verify the password.");
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
