// Shared, desktop-safe confirm (sc-11968).
//
// `window.confirm` is unreliable inside the Tauri desktop WebView: it can silently
// no-op and return `undefined` rather than a boolean, so a raw confirm there can
// neither confirm nor cancel — a "Discard?" guard built on it is dead on desktop. This
// module replaces it with a promise-returning confirm backed by a REAL in-app React
// dialog (the shared `Modal`, portaled to <body>), so the same code path works in the
// browser and the desktop shell.
//
// Usage:
//   import { appConfirm, useConfirm, ConfirmHost } from "../appConfirm.jsx";
//   // once, near the app root:  <ConfirmHost />
//   if (await appConfirm({ message, tone: "danger" })) { ...discard... }
//
// `appConfirm` is a plain function (callable from anywhere — sync or async code, event
// handlers, or a leave-guard callback), and `useConfirm()` is a thin React accessor that
// returns it. Both resolve to `true` (proceed) or `false` (cancel).
//
// Adopters: the Image Editor Close/Discard + its leave-guard use this today. The other
// window.confirm call sites (PresetManager, Training, Pose Library, the video EditorScreen
// timeline-switch guard, App's trash-unavailable confirm) should migrate to it too so the
// whole app is desktop-safe — tracked as follow-ups on sc-11968's siblings (S10–S12).
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Modal } from "./components/Modal.jsx";

// Module-level bridge to the mounted <ConfirmHost/>. The host installs its
// `open(options) => Promise<boolean>` here on mount and clears it on unmount; appConfirm
// delegates to it. A singleton (not context) so non-React / imperative callers — e.g. a
// leave-guard callback handed to App — can reach the dialog without threading a hook.
let hostOpen = null;

// Normalize the caller's options — a bare string is shorthand for `{ message }`.
export function normalizeConfirmOptions(options) {
  const opts = typeof options === "string" ? { message: options } : options ?? {};
  return {
    title: opts.title ?? "Are you sure?",
    message: opts.message ?? "",
    confirmLabel: opts.confirmLabel ?? "Confirm",
    cancelLabel: opts.cancelLabel ?? "Cancel",
    // "danger" styles the confirm button as destructive (discard/leave/delete).
    tone: opts.tone === "danger" ? "danger" : "default",
  };
}

// Desktop-safe confirm. Resolves true (proceed) / false (cancel). When a <ConfirmHost/>
// is mounted (always, in the real app) it renders a React dialog; otherwise it falls back
// to the platform confirm, preserving the app's long-standing "no confirm available →
// proceed" semantics for non-DOM / test contexts that didn't mount the host.
export function appConfirm(options = {}) {
  const opts = normalizeConfirmOptions(options);
  if (typeof hostOpen === "function") {
    return hostOpen(opts);
  }
  const ok =
    typeof window !== "undefined" && typeof window.confirm === "function"
      ? window.confirm(opts.message)
      : true;
  return Promise.resolve(Boolean(ok));
}

// React-idiomatic accessor: `const confirm = useConfirm(); await confirm({ ... })`.
// Returns the stable module function, so it never busts a dependency array.
export function useConfirm() {
  return appConfirm;
}

// Mount ONCE near the app root. Renders the confirm dialog while a request is pending and
// registers its opener into the module bridge so appConfirm()/useConfirm() resolve through
// a real dialog. Escape, a backdrop click, or unmount all settle the promise as cancelled,
// so an awaiter is never left hanging.
export function ConfirmHost() {
  const [request, setRequest] = useState(null);
  const pendingRef = useRef(null);

  const settle = useCallback((result) => {
    const resolve = pendingRef.current;
    pendingRef.current = null;
    setRequest(null);
    if (resolve) resolve(result);
  }, []);

  const open = useCallback((options) => {
    return new Promise((resolve) => {
      // A second request while one is still open cancels the first, so its awaiter never
      // hangs, then supersedes it with the new dialog.
      if (pendingRef.current) {
        const previous = pendingRef.current;
        pendingRef.current = null;
        previous(false);
      }
      pendingRef.current = resolve;
      setRequest(options);
    });
  }, []);

  useEffect(() => {
    hostOpen = open;
    return () => {
      if (hostOpen === open) hostOpen = null;
      if (pendingRef.current) {
        const resolve = pendingRef.current;
        pendingRef.current = null;
        resolve(false);
      }
    };
  }, [open]);

  if (!request) return null;

  return (
    <Modal
      className="discard-confirm-modal app-confirm-modal"
      labelledBy="app-confirm-title"
      onClose={() => settle(false)}
    >
      <h2 className="discard-confirm-title" id="app-confirm-title">
        {request.title}
      </h2>
      {request.message ? <p className="discard-confirm-body">{request.message}</p> : null}
      <div className="discard-confirm-actions">
        <button onClick={() => settle(false)} type="button">
          {request.cancelLabel}
        </button>
        <button
          className={request.tone === "danger" ? "danger-action" : "primary-action"}
          onClick={() => settle(true)}
          type="button"
        >
          {request.confirmLabel}
        </button>
      </div>
    </Modal>
  );
}
