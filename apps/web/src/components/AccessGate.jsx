import React, { useEffect, useRef } from "react";
import { Logo } from "./Logo.jsx";

// Full-page remote-access blocker (epic 4484 story 7, hardened in sc-15102).
//
// This is rendered INSTEAD of the app shell — App early-returns it, so no sidebar,
// no topbar, no screens exist while it is up. It replaced an `.auth-band` banner
// that sat above a fully live shell: because every protected data load is held
// until `authenticated`, a remote browser could navigate the whole nav tree and
// find nothing but empty screens, and the one-line password field at the top read
// as an optional extra rather than the thing standing between it and the app.
//
// Three blocking states, chosen by App from the hook's `gateStatus`:
//   locked        — the host requires a password and we don't have a good one.
//   unlocking     — a password restored from localStorage is being re-verified
//                   against the host (it may have been changed or cleared since).
//   awaiting-host — the /api/v1/access probe has failed, so we don't yet know
//                   whether a password is required. Never shown for the first
//                   in-flight probe, only once one has failed, so a healthy host
//                   does not flash this on every load.
//
// `notices` renders here too: the gate owns the whole viewport, so the access-related
// messages have nowhere else to surface while it is up. Only those kinds are shown —
// the gate is a lock screen, not the app's error console, and the rest of the store is
// full of screen-level failures (e.g. the health poll's bare "Failed to fetch") that
// tell a locked-out visitor nothing they can act on.
const GATE_NOTICE_KINDS = new Set(["access-probe", "access-verify", "media-ticket"]);
export function AccessGate({
  authError,
  notices = [],
  onForget,
  onSubmit,
  passwordDraft,
  setPasswordDraft,
  status,
}) {
  const inputRef = useRef(null);
  // Focus the password box whenever it becomes the live control — including when the
  // gate transitions out of unlocking/awaiting-host on a retry, which `autoFocus`
  // (mount-only) would miss because the gate itself is already mounted.
  useEffect(() => {
    if (status === "locked") {
      inputRef.current?.focus();
    }
  }, [status]);

  return (
    <main className="access-gate">
      <div className="access-gate-card">
        <span className="access-gate-mark" aria-hidden="true">
          <Logo size={52} />
        </span>
        <h1>SceneWorks</h1>

        {status === "locked" ? (
          <>
            <p className="access-gate-lede">
              This host is password protected. Enter its access password to continue.
            </p>
            <form className="access-gate-form" onSubmit={onSubmit}>
              <label htmlFor="token">Password</label>
              <input
                autoComplete="current-password"
                id="token"
                onChange={(event) => setPasswordDraft(event.target.value)}
                placeholder="Enter the access password"
                ref={inputRef}
                type="password"
                value={passwordDraft}
              />
              <button className="access-gate-cta" type="submit">
                Unlock
              </button>
            </form>
            {/* Wrong-password / rejected-saved-password feedback belongs against the
                control that produced it, above the standing hint. */}
            {authError ? (
              <p className="notice error access-gate-error" role="alert">
                {authError}
              </p>
            ) : null}
            <p className="access-gate-hint">
              The password is set on the host machine, under Settings → Remote access.
            </p>
          </>
        ) : (
          <>
            <p className="access-gate-lede">
              {status === "unlocking"
                ? "Checking the saved password with the host…"
                : "Contacting the host to check whether a password is required…"}
            </p>
            <p className="access-gate-status" role="status">
              Retrying automatically.
            </p>
            {/* Escape hatch: without it, a saved password that can't be verified (host
                down, or the host cleared the check) is a dead end with no way back to
                the prompt. */}
            {status === "unlocking" && onForget ? (
              <button className="access-gate-secondary" onClick={onForget} type="button">
                Use a different password
              </button>
            ) : null}
          </>
        )}

        {notices
          .filter((notice) => GATE_NOTICE_KINDS.has(notice.kind))
          .map((notice) => (
            <p className="notice error" key={notice.kind}>
              {notice.message}
            </p>
          ))}
      </div>
    </main>
  );
}
